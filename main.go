package main

import (
	"bufio"
	"bytes"
	"container/list"
	"crypto/md5"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// HttpError represents an error with HTTP status
type HttpError struct {
	Message string
	Status  int
}

func (e *HttpError) Error() string {
	return e.Message
}

const (
	BaseUrl                = "https://generativelanguage.googleapis.com"
	ApiVersion             = "v1beta"
	ApiClient              = "genai-js/0.21.0"
	DefaultModel           = "gemini-1.5-pro-latest"
	DefaultEmbeddingsModel = "text-embedding-004"
)

type cacheEntry struct {
	key        string
	insertTime int64 // 使用纳秒时间戳提高比较效率
	listElem   *list.Element
}

type Cache struct {
	mu      sync.Mutex
	data    map[string]*cacheEntry
	list    *list.List // 维护插入顺序（链表头为最旧元素）
	maxSize int
}

func NewCache(maxSize int) *Cache {
	return &Cache{
		data:    make(map[string]*cacheEntry),
		list:    list.New(),
		maxSize: maxSize,
	}
}

var (
	keyCache     *Cache
	storeKeyList []string
	proxyCIDR    string
	authKey      string
)

func InitEnv(configPrefix string) {
	// If no API keys in header, try to get from environment
	envTokens := os.Getenv(configPrefix + "TOKENS")
	if envTokens != "" {
		apiKeys := strings.Split(envTokens, ",")
		storeKeyList = append(storeKeyList, apiKeys...)
	}
	keyCache = NewCache(1_000_000)
	proxyCIDR = os.Getenv(configPrefix + "PROXY_CIDR")
	authKey = os.Getenv(configPrefix + "AUTH_KEY")
}

func (c *Cache) selectLessUseKey(keys []string) string {
	c.mu.Lock()
	defer c.mu.Unlock()

	var (
		candidateKeys []string
		existingKeys  []string
		oldestKey     string
		oldestTime    int64 = -1
	)

	for _, key := range keys {
		if entry, exists := c.data[key]; exists {
			existingKeys = append(existingKeys, key)
			if oldestTime == -1 || entry.insertTime < oldestTime {
				oldestTime = entry.insertTime
				oldestKey = key
			}
		} else {
			candidateKeys = append(candidateKeys, key)
		}
	}

	if len(candidateKeys) > 0 {
		useKey := candidateKeys[rand.Intn(len(candidateKeys))]
		c.Add(useKey)
		return useKey
	}

	c.Add(oldestKey)
	return oldestKey
}

func (c *Cache) Add(key string) {
	// 存在性检查
	if entry, exists := c.data[key]; exists {
		// 移动现有元素到链表尾部表示最近访问
		c.list.MoveToBack(entry.listElem)
		entry.insertTime = time.Now().UnixNano()
		return
	}

	// 新增元素时的淘汰逻辑
	if c.list.Len() >= c.maxSize {
		// 淘汰策略：移除最旧元素
		if oldestElem := c.list.Front(); oldestElem != nil {
			delete(c.data, oldestElem.Value.(string))
			c.list.Remove(oldestElem)
		}
	}

	// 添加新元素到链表尾部
	elem := c.list.PushBack(key)
	c.data[key] = &cacheEntry{
		key:        key,
		insertTime: time.Now().UnixNano(),
		listElem:   elem,
	}
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
	// Handle CORS preflight request
	if r.Method == "OPTIONS" {
		handleOPTIONS(w)
		return
	}

	// Extract the path from the URL
	path := r.URL.Path

	// Get authorization tokens
	apiKeys, err := getAPIKeys(r)
	if err != nil {
		handleError(w, err)
		return
	}

	// Select a random API key
	apiKey := keyCache.selectLessUseKey(apiKeys)
	log.Printf("use : %s", apiKey)

	// Route to appropriate handler based on path
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		if r.Method != "POST" {
			handleError(w, &HttpError{"Method not allowed", http.StatusMethodNotAllowed})
			return
		}
		handleCompletions(w, r, apiKey)
	case strings.HasSuffix(path, "/embeddings"):
		if r.Method != "POST" {
			handleError(w, &HttpError{"Method not allowed", http.StatusMethodNotAllowed})
			return
		}
		handleEmbeddings(w, r, apiKey)
	case strings.HasSuffix(path, "/models"):
		if r.Method != "GET" {
			handleError(w, &HttpError{"Method not allowed", http.StatusMethodNotAllowed})
			return
		}
		handleModels(w, apiKey)
	default:
		handleError(w, &HttpError{"404 Not Found", http.StatusNotFound})
	}
}

func getAPIKeys(r *http.Request) ([]string, error) {
	// Get API keys from Authorization header
	auth := r.Header.Get("Authorization")
	if auth != "" {
		if auth == authKey {
			if len(storeKeyList) > 0 {
				return storeKeyList, nil
			}
		} else {
			parts := strings.Split(auth, " ")
			if len(parts) == 2 {
				apiKeys := strings.Split(parts[1], ",")
				if len(apiKeys) > 0 {
					return apiKeys, nil
				}
			}
		}
	}

	// If no API keys found, return error
	return nil, &HttpError{"403 No Auth", http.StatusForbidden}
}

func handleOPTIONS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "*")
	w.Header().Set("Access-Control-Allow-Headers", "*")
	w.WriteHeader(http.StatusOK)
}

func handleError(w http.ResponseWriter, err error) {
	httpErr, ok := err.(*HttpError)
	if !ok {
		httpErr = &HttpError{err.Error(), http.StatusInternalServerError}
	}

	log.Printf("Error: %s", httpErr.Error())

	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(httpErr.Status)
	w.Write([]byte(httpErr.Message))
}

func fixCors(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
}

func makeHeaders(apiKey string, moreHeaders map[string]string) map[string]string {
	headers := map[string]string{
		"x-goog-api-client": ApiClient,
	}

	if apiKey != "" {
		headers["x-goog-api-key"] = apiKey
	}

	// Add additional headers
	for k, v := range moreHeaders {
		headers[k] = v
	}

	return headers
}

// Calculate proxy URL from token using MD5->BigInt and CIDR calculation
func getProxyIpv6EndPoint(token string) (string, error) {
	if proxyCIDR == "" {
		return "", nil // If any proxy setting is missing, don't use proxy
	}

	// Calculate MD5 hash of token
	hash := md5.Sum([]byte(token))

	// Convert MD5 hash to bigint
	bigIntValue := new(big.Int).SetBytes(hash[:])

	// Parse CIDR
	_, ipNet, err := net.ParseCIDR(proxyCIDR)
	if err != nil {
		return "", fmt.Errorf("invalid proxy_cidr: %v", err)
	}

	if ipNet.IP.To4() != nil {
		return "", fmt.Errorf("IPv6 CIDR required, got IPv4: %s", proxyCIDR)
	}

	// Calculate the number of addresses in the CIDR block
	ones, bits := ipNet.Mask.Size()
	addrCount := new(big.Int).Lsh(big.NewInt(1), uint(bits-ones))

	// Subtract 20 (10 for the beginning, 10 for the end)
	addrCount.Sub(addrCount, big.NewInt(20))

	if addrCount.Cmp(big.NewInt(0)) != 1 {
		return "", fmt.Errorf("not enough proxy_cidr: %v", err)
	}

	// Take modulo of the hash value
	index := new(big.Int).Mod(bigIntValue, addrCount)

	// Add 10 to skip the first 10 addresses
	index.Add(index, big.NewInt(10))

	// Convert to IPv6 address
	ip := generateIPv6(ipNet, index)

	return ip.String(), nil
}

// Generate IPv6 address from CIDR and index
func generateIPv6(ipNet *net.IPNet, index *big.Int) net.IP {
	// Get the first IP in the range
	ip := ipNet.IP.To16()

	// Create a new big.Int from the IP
	ipInt := new(big.Int).SetBytes(ip)

	// Add the index
	ipInt.Add(ipInt, index)

	// Convert back to IP
	ipBytes := ipInt.Bytes()

	// Ensure we have 16 bytes for IPv6
	result := make([]byte, 16)
	copy(result[16-len(ipBytes):], ipBytes)

	return result
}

// Create an HTTP client with proxy
func createHTTPClient(token string) (*http.Client, error) {
	endPointIP, err := getProxyIpv6EndPoint(token)
	if err != nil {
		return nil, err
	}

	log.Printf("use : %s, ipv6: %s", token, endPointIP)

	if endPointIP != "" {
		dialer := &net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 1200 * time.Second,
			LocalAddr: &net.TCPAddr{
				IP:   net.ParseIP(endPointIP),
				Port: 0,
			},
		}

		transport := &http.Transport{
			Proxy:       http.ProxyFromEnvironment,
			DialContext: dialer.DialContext,
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		}

		return &http.Client{
			Transport: transport,
			Timeout:   30 * time.Second,
		}, nil
	}

	return &http.Client{
		Timeout: time.Second * 30,
	}, nil
}

// Models response struct
type ModelsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// OpenAI-compatible models response
type OpenAIModelsResponse struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

type OpenAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func handleModels(w http.ResponseWriter, apiKey string) {
	client, err := createHTTPClient(apiKey)
	if err != nil {
		handleError(w, err)
		return
	}

	modelUrl := fmt.Sprintf("%s/%s/models", BaseUrl, ApiVersion)

	req, err := http.NewRequest("GET", modelUrl, nil)
	if err != nil {
		handleError(w, err)
		return
	}

	headers := makeHeaders(apiKey, nil)
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		handleError(w, err)
		return
	}
	defer resp.Body.Close()

	// Copy headers from the API response
	for k, v := range resp.Header {
		for _, val := range v {
			w.Header().Add(k, val)
		}
	}

	// Set CORS headers
	fixCors(w)

	if resp.StatusCode != http.StatusOK {
		// Copy the body from the API response
		io.Copy(w, resp.Body)
		return
	}

	// Parse response
	var modelsResp ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		handleError(w, err)
		return
	}

	// Transform to OpenAI format
	openAIResp := OpenAIModelsResponse{
		Object: "list",
		Data:   make([]OpenAIModel, 0, len(modelsResp.Models)),
	}

	for _, model := range modelsResp.Models {
		openAIResp.Data = append(openAIResp.Data, OpenAIModel{
			ID:      strings.Replace(model.Name, "models/", "", 1),
			Object:  "model",
			Created: 0,
			OwnedBy: "",
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(openAIResp)
}

// Embeddings request struct
type EmbeddingsRequest struct {
	Model      string      `json:"model"`
	Input      interface{} `json:"input"`
	Dimensions int         `json:"dimensions,omitempty"`
}

// Gemini embeddings request
type GeminiEmbeddingsRequest struct {
	Requests []GeminiEmbeddingRequest `json:"requests"`
}

type GeminiEmbeddingRequest struct {
	Model                string                 `json:"model"`
	Content              GeminiEmbeddingContent `json:"content"`
	OutputDimensionality int                    `json:"outputDimensionality,omitempty"`
}

type GeminiEmbeddingContent struct {
	Parts GeminiEmbeddingParts `json:"parts"`
}

type GeminiEmbeddingParts struct {
	Text string `json:"text"`
}

// Gemini embeddings response
type GeminiEmbeddingsResponse struct {
	Embeddings []GeminiEmbedding `json:"embeddings"`
}

type GeminiEmbedding struct {
	Values []float64 `json:"values"`
}

// OpenAIEmbeddingsResponse OpenAI-compatible embeddings response
type OpenAIEmbeddingsResponse struct {
	Object string            `json:"object"`
	Data   []OpenAIEmbedding `json:"data"`
	Model  string            `json:"model"`
}

type OpenAIEmbedding struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request, apiKey string) {
	// Parse request
	var req EmbeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		handleError(w, err)
		return
	}

	// Check model
	if req.Model == "" {
		handleError(w, &HttpError{"model is not specified", http.StatusBadRequest})
		return
	}

	// Convert input to array if it's not already
	var inputArray []string
	switch v := req.Input.(type) {
	case string:
		inputArray = []string{v}
	case []interface{}:
		inputArray = make([]string, len(v))
		for i, item := range v {
			inputArray[i] = fmt.Sprintf("%v", item)
		}
	case []string:
		inputArray = v
	default:
		// Try to convert to string
		inputArray = []string{fmt.Sprintf("%v", req.Input)}
	}

	// Determine model name
	model := req.Model
	if !strings.HasPrefix(model, "models/") {
		model = "models/" + DefaultEmbeddingsModel
		req.Model = DefaultEmbeddingsModel
	}

	// Create API request
	geminiReq := GeminiEmbeddingsRequest{
		Requests: make([]GeminiEmbeddingRequest, 0, len(inputArray)),
	}

	for _, text := range inputArray {
		geminiReq.Requests = append(geminiReq.Requests, GeminiEmbeddingRequest{
			Model: model,
			Content: GeminiEmbeddingContent{
				Parts: GeminiEmbeddingParts{
					Text: text,
				},
			},
			OutputDimensionality: req.Dimensions,
		})
	}

	reqBody, err := json.Marshal(geminiReq)
	if err != nil {
		handleError(w, err)
		return
	}

	// Create HTTP client
	client, err := createHTTPClient(apiKey)
	if err != nil {
		handleError(w, err)
		return
	}

	// Create request
	url := fmt.Sprintf("%s/%s/%s:batchEmbedContents", BaseUrl, ApiVersion, model)
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(reqBody))
	if err != nil {
		handleError(w, err)
		return
	}

	// Add headers
	headers := makeHeaders(apiKey, map[string]string{"Content-Type": "application/json"})
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}

	// Send request
	resp, err := client.Do(httpReq)
	if err != nil {
		handleError(w, err)
		return
	}
	defer resp.Body.Close()

	// Copy headers
	for k, v := range resp.Header {
		for _, val := range v {
			w.Header().Add(k, val)
		}
	}

	// Set CORS headers
	fixCors(w)

	if resp.StatusCode != http.StatusOK {
		// Copy the error response
		io.Copy(w, resp.Body)
		return
	}

	// Parse response
	var geminiResp GeminiEmbeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		handleError(w, err)
		return
	}

	// Transform to OpenAI format
	openAIResp := OpenAIEmbeddingsResponse{
		Object: "list",
		Data:   make([]OpenAIEmbedding, 0, len(geminiResp.Embeddings)),
		Model:  req.Model,
	}

	for i, embedding := range geminiResp.Embeddings {
		openAIResp.Data = append(openAIResp.Data, OpenAIEmbedding{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding.Values,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(openAIResp)
}

// Chat completions request struct
type ChatCompletionsRequest struct {
	Model          string          `json:"model"`
	Messages       []ChatMessage   `json:"messages"`
	Stream         bool            `json:"stream"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Temperature    float64         `json:"temperature,omitempty"`
	TopP           float64         `json:"top_p,omitempty"`
	TopK           int             `json:"top_k,omitempty"`
	N              int             `json:"n,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	StreamOptions  *StreamOptions  `json:"stream_options,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// Gemini request structs
type GeminiRequest struct {
	Contents          []GeminiContent        `json:"contents,omitempty"`
	SystemInstruction *GeminiContent         `json:"systemInstruction,omitempty"`
	SafetySettings    []GeminiSafetySetting  `json:"safetySettings"`
	GenerationConfig  GeminiGenerationConfig `json:"generationConfig"`
}

type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *GeminiInlineData `json:"inlineData,omitempty"`
}

type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type GeminiSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type GeminiGenerationConfig struct {
	StopSequences    []string        `json:"stopSequences,omitempty"`
	CandidateCount   int             `json:"candidateCount,omitempty"`
	MaxOutputTokens  int             `json:"maxOutputTokens,omitempty"`
	Temperature      float64         `json:"temperature,omitempty"`
	TopP             float64         `json:"topP,omitempty"`
	TopK             int             `json:"topK,omitempty"`
	FrequencyPenalty float64         `json:"frequencyPenalty,omitempty"`
	PresencePenalty  float64         `json:"presencePenalty,omitempty"`
	ResponseMimeType string          `json:"responseMimeType,omitempty"`
	ResponseSchema   json.RawMessage `json:"responseSchema,omitempty"`
}

// Gemini response structs
type GeminiResponse struct {
	Candidates    []GeminiCandidate `json:"candidates"`
	UsageMetadata GeminiUsage       `json:"usageMetadata"`
}

type GeminiCandidate struct {
	Index        int           `json:"index"`
	Content      GeminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type GeminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// OpenAI-compatible response structs
type OpenAIChatCompletion struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int            `json:"index"`
	Message      *OpenAIMessage `json:"message,omitempty"`
	Delta        *OpenAIDelta   `json:"delta,omitempty"`
	FinishReason string         `json:"finish_reason"`
	Logprobs     interface{}    `json:"logprobs"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Safety settings
var harmCategories = []string{
	"HARM_CATEGORY_HATE_SPEECH",
	"HARM_CATEGORY_SEXUALLY_EXPLICIT",
	"HARM_CATEGORY_DANGEROUS_CONTENT",
	"HARM_CATEGORY_HARASSMENT",
	"HARM_CATEGORY_CIVIC_INTEGRITY",
}

var safetySettings = func() []GeminiSafetySetting {
	settings := make([]GeminiSafetySetting, 0, len(harmCategories))
	for _, category := range harmCategories {
		settings = append(settings, GeminiSafetySetting{
			Category:  category,
			Threshold: "BLOCK_NONE",
		})
	}
	return settings
}()

// Fields mapping
var fieldsMap = map[string]string{
	"stop":                  "stopSequences",
	"n":                     "candidateCount",
	"max_tokens":            "maxOutputTokens",
	"max_completion_tokens": "maxOutputTokens",
	"temperature":           "temperature",
	"top_p":                 "topP",
	"top_k":                 "topK",
	"frequency_penalty":     "frequencyPenalty",
	"presence_penalty":      "presencePenalty",
}

// Reason mapping
var reasonsMap = map[string]string{
	"STOP":       "stop",
	"MAX_TOKENS": "length",
	"SAFETY":     "content_filter",
	"RECITATION": "content_filter",
}

// Generate a random chat completion ID
func generateChatcmplId() string {
	const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	rand.Seed(time.Now().UnixNano())

	result := "chatcmpl-"
	for i := 0; i < 29; i++ {
		result += string(charset[rand.Intn(len(charset))])
	}

	return result
}

// Transform config
func transformConfig(req *ChatCompletionsRequest) GeminiGenerationConfig {
	config := GeminiGenerationConfig{}

	// Apply fields from the map
	if req.MaxTokens > 0 {
		config.MaxOutputTokens = req.MaxTokens
	}

	if req.Temperature > 0 {
		config.Temperature = req.Temperature
	}

	if req.TopP > 0 {
		config.TopP = req.TopP
	}

	if req.TopK > 0 {
		config.TopK = req.TopK
	}

	if req.N > 0 {
		config.CandidateCount = req.N
	}

	if len(req.Stop) > 0 {
		config.StopSequences = req.Stop
	}

	// Handle response format
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_schema":
			config.ResponseSchema = req.ResponseFormat.JSONSchema
			if len(req.ResponseFormat.JSONSchema) > 0 {
				var schema map[string]interface{}
				if err := json.Unmarshal(req.ResponseFormat.JSONSchema, &schema); err == nil {
					if _, ok := schema["enum"]; ok {
						config.ResponseMimeType = "text/x.enum"
						break
					}
				}
			}
			fallthrough
		case "json_object":
			config.ResponseMimeType = "application/json"
		case "text":
			config.ResponseMimeType = "text/plain"
		}
	}

	return config
}

// Parse image from URL or data URI
func parseImg(imgURL string) (*GeminiInlineData, error) {
	if strings.HasPrefix(imgURL, "http://") || strings.HasPrefix(imgURL, "https://") {
		// Fetch image from URL
		resp, err := http.Get(imgURL)
		if err != nil {
			return nil, fmt.Errorf("error fetching image: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("error fetching image: %d %s (%s)",
				resp.StatusCode, resp.Status, imgURL)
		}

		// Read and encode image
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("error reading image: %v", err)
		}

		return &GeminiInlineData{
			MimeType: resp.Header.Get("Content-Type"),
			Data:     base64.StdEncoding.EncodeToString(data),
		}, nil
	} else if strings.HasPrefix(imgURL, "data:") {
		// Parse data URI
		parts := strings.SplitN(imgURL, ",", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid image data URI")
		}

		mimeTypePart := strings.SplitN(parts[0], ";", 2)
		mimeType := strings.TrimPrefix(mimeTypePart[0], "data:")

		var data string
		if len(mimeTypePart) > 1 && mimeTypePart[1] == "base64" {
			data = parts[1]
		} else {
			// Not base64 encoded, need to encode
			data = base64.StdEncoding.EncodeToString([]byte(parts[1]))
		}

		return &GeminiInlineData{
			MimeType: mimeType,
			Data:     data,
		}, nil
	}

	return nil, fmt.Errorf("unsupported image format")
}

// Transform message
func transformMsg(msg ChatMessage) (GeminiContent, error) {
	content := GeminiContent{
		Parts: []GeminiPart{},
	}

	// Convert role
	switch msg.Role {
	case "system":
		content.Role = msg.Role
	case "assistant":
		content.Role = "model"
	default:
		content.Role = "user"
	}

	// Process content
	switch c := msg.Content.(type) {
	case string:
		// Simple text content
		content.Parts = append(content.Parts, GeminiPart{Text: c})
	case []interface{}:
		// Array of content parts
		for _, item := range c {
			if part, ok := item.(map[string]interface{}); ok {
				switch part["type"] {
				case "text":
					if text, ok := part["text"].(string); ok {
						content.Parts = append(content.Parts, GeminiPart{Text: text})
					}
				case "image_url":
					if urlMap, ok := part["image_url"].(map[string]interface{}); ok {
						if url, ok := urlMap["url"].(string); ok {
							imgData, err := parseImg(url)
							if err != nil {
								return content, err
							}
							content.Parts = append(content.Parts, GeminiPart{InlineData: imgData})
						}
					}
				case "input_audio":
					if audio, ok := part["input_audio"].(map[string]interface{}); ok {
						format, _ := audio["format"].(string)
						data, _ := audio["data"].(string)

						content.Parts = append(content.Parts, GeminiPart{
							InlineData: &GeminiInlineData{
								MimeType: "audio/" + format,
								Data:     data,
							},
						})
					}
				}
			}
		}

		// Add empty text if only images to avoid API error
		if len(content.Parts) > 0 {
			allImages := true
			for _, part := range content.Parts {
				if part.Text != "" {
					allImages = false
					break
				}
			}
			if allImages {
				content.Parts = append(content.Parts, GeminiPart{Text: ""})
			}
		}
	default:
		// Try to convert to string
		content.Parts = append(content.Parts, GeminiPart{Text: fmt.Sprintf("%v", msg.Content)})
	}

	return content, nil
}

// Transform messages
func transformMessages(messages []ChatMessage) ([]GeminiContent, *GeminiContent, error) {
	var contents []GeminiContent
	var systemInstruction *GeminiContent

	for _, msg := range messages {
		content, err := transformMsg(msg)
		if err != nil {
			return nil, nil, err
		}

		if msg.Role == "system" {
			systemInstr := content
			systemInstruction = &systemInstr
		} else {
			contents = append(contents, content)
		}
	}

	// Add empty model message if only system instruction
	if systemInstruction != nil && len(contents) == 0 {
		contents = append(contents, GeminiContent{
			Role:  "model",
			Parts: []GeminiPart{{Text: " "}},
		})
	}

	return contents, systemInstruction, nil
}

// Transform request
func transformRequest(req *ChatCompletionsRequest) (*GeminiRequest, error) {
	contents, systemInstruction, err := transformMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	geminiReq := &GeminiRequest{
		Contents:          contents,
		SystemInstruction: systemInstruction,
		SafetySettings:    safetySettings,
		GenerationConfig:  transformConfig(req),
	}

	return geminiReq, nil
}

// Transform candidate to OpenAI choice
func transformCandidate(candidate GeminiCandidate, isStream bool) OpenAIChoice {
	// Join text parts
	content := ""
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			if content != "" {
				content += "\n\n|>"
			}
			content += part.Text
		}
	}

	choice := OpenAIChoice{
		Index:        candidate.Index,
		FinishReason: reasonsMap[candidate.FinishReason],
		Logprobs:     nil,
	}

	if isStream {
		choice.Delta = &OpenAIDelta{
			Role:    "assistant",
			Content: content,
		}
	} else {
		choice.Message = &OpenAIMessage{
			Role:    "assistant",
			Content: content,
		}
	}

	if choice.FinishReason == "" {
		choice.FinishReason = candidate.FinishReason
	}

	return choice
}

// Process completions response
func processCompletionsResponse(data *GeminiResponse, model string, id string) (*OpenAIChatCompletion, error) {
	choices := make([]OpenAIChoice, 0, len(data.Candidates))

	for _, candidate := range data.Candidates {
		choices = append(choices, transformCandidate(candidate, false))
	}

	response := &OpenAIChatCompletion{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: choices,
		Usage: &OpenAIUsage{
			PromptTokens:     data.UsageMetadata.PromptTokenCount,
			CompletionTokens: data.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      data.UsageMetadata.TotalTokenCount,
		},
	}

	return response, nil
}

// Handle non-streaming completions
func handleNonStreamingCompletions(w http.ResponseWriter, geminiReq *GeminiRequest, apiKey, model, url string) {
	// Convert request to JSON
	reqBody, err := json.Marshal(geminiReq)
	if err != nil {
		handleError(w, err)
		return
	}

	// Create HTTP client
	client, err := createHTTPClient(apiKey)
	if err != nil {
		handleError(w, err)
		return
	}

	// Create HTTP request
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(reqBody))
	if err != nil {
		handleError(w, err)
		return
	}

	// Add headers
	headers := makeHeaders(apiKey, map[string]string{"Content-Type": "application/json"})
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}

	// Send request
	resp, err := client.Do(httpReq)
	if err != nil {
		handleError(w, err)
		return
	}
	defer resp.Body.Close()

	// Copy headers
	for k, v := range resp.Header {
		for _, val := range v {
			w.Header().Add(k, val)
		}
	}

	// Set CORS headers
	fixCors(w)

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		// Read and write error response
		errBody, _ := io.ReadAll(resp.Body)
		w.Write(errBody)
		log.Printf("use : %s, err: %s", authKey, string(errBody))
		return
	}

	// Parse response
	var geminiResp GeminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		handleError(w, err)
		return
	}

	// Generate chat completion ID
	id := generateChatcmplId()

	// Transform to OpenAI format
	openAIResp, err := processCompletionsResponse(&geminiResp, model, id)
	if err != nil {
		handleError(w, err)
		return
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(openAIResp)
}

// Handle streaming completions
func handleStreamingCompletions(w http.ResponseWriter, req *ChatCompletionsRequest, geminiReq *GeminiRequest, apiKey, model, url string) {
	// Set headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	fixCors(w)
	w.WriteHeader(http.StatusOK)

	// Flush headers
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	// Convert request to JSON
	reqBody, err := json.Marshal(geminiReq)
	if err != nil {
		fmt.Fprintf(w, "data: %s\n\n", err.Error())
		return
	}

	// Create HTTP client
	client, err := createHTTPClient(apiKey)
	if err != nil {
		fmt.Fprintf(w, "data: %s\n\n", err.Error())
		return
	}

	// Create HTTP request
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(reqBody))
	if err != nil {
		fmt.Fprintf(w, "data: %s\n\n", err.Error())
		return
	}

	// Add headers
	headers := makeHeaders(apiKey, map[string]string{"Content-Type": "application/json"})
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}

	// Send request
	resp, err := client.Do(httpReq)
	if err != nil {
		fmt.Fprintf(w, "data: %s\n\n", err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		// Read and write error response
		errBody, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(w, "data: %s\n\n", string(errBody))
		log.Printf("use : %s, err: %s", authKey, string(errBody))
		return
	}

	// Generate chat completion ID
	id := generateChatcmplId()

	// Process the stream
	scanner := bufio.NewScanner(resp.Body)
	isFirst := true
	var lastChoice *OpenAIChoice
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			break
		}

		var geminiResp GeminiResponse
		if err := json.Unmarshal([]byte(data), &geminiResp); err != nil {
			continue
		}

		if len(geminiResp.Candidates) == 0 {
			continue
		}

		candidate := geminiResp.Candidates[0]
		choice := transformCandidate(candidate, true)

		if isFirst {
			isFirst = false
			// First response always includes role
		} else if choice.Delta != nil {
			// Subsequent responses don't include role
			choice.Delta.Role = ""
		}

		if lastChoice != nil && choice.Delta != nil && choice.Delta.Content == "" {
			// Skip empty content updates
			continue
		}

		response := OpenAIChatCompletion{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []OpenAIChoice{choice},
		}

		// Include usage if requested
		if includeUsage && candidate.FinishReason != "" {
			response.Usage = &OpenAIUsage{
				PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
				CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
			}
		}

		// Write the response
		respBytes, err := json.Marshal(response)
		if err != nil {
			continue
		}

		fmt.Fprintf(w, "data: %s\n\n", string(respBytes))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		lastChoice = &choice
	}

	// Send final message with finish_reason
	if lastChoice != nil {
		lastChoice.FinishReason = "stop"
		lastChoice.Delta = &OpenAIDelta{}

		response := OpenAIChatCompletion{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []OpenAIChoice{*lastChoice},
		}

		respBytes, _ := json.Marshal(response)
		fmt.Fprintf(w, "data: %s\n\n", string(respBytes))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
}

func handleCompletions(w http.ResponseWriter, r *http.Request, apiKey string) {
	// Parse request
	var req ChatCompletionsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		handleError(w, &HttpError{fmt.Sprintf("Failed to parse request: %v", err), http.StatusBadRequest})
		return
	}

	// Set default model if not specified
	model := DefaultModel
	if req.Model != "" {
		if strings.HasPrefix(req.Model, "models/") {
			model = req.Model[7:] // Remove "models/" prefix
		} else if strings.HasPrefix(req.Model, "gemini-") || strings.HasPrefix(req.Model, "learnlm-") {
			model = req.Model
		}
	}

	// Transform request to Gemini format
	geminiReq, err := transformRequest(&req)
	if err != nil {
		handleError(w, err)
		return
	}

	// Build URL based on stream option
	task := "generateContent"
	url := fmt.Sprintf("%s/%s/models/%s:%s", BaseUrl, ApiVersion, model, task)

	if req.Stream {
		task = "streamGenerateContent"
		url = fmt.Sprintf("%s/%s/models/%s:%s?alt=sse", BaseUrl, ApiVersion, model, task)
		handleStreamingCompletions(w, &req, geminiReq, apiKey, model, url)
	} else {
		handleNonStreamingCompletions(w, geminiReq, apiKey, model, url)
	}
}

func main() {
	configPrefix := "GEMINI2OAI_"
	port := os.Getenv(configPrefix + "PORT")
	if port == "" {
		port = "3000"
	}

	InitEnv(configPrefix)

	http.HandleFunc("/", handleRequest)
	log.Printf("Server listening on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
