#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 1. The Configuration (Proves you read the Llama/GPT paper)
typedef struct {
    int dim;        // Transformer dimension
    int hidden_dim; // FFN layer dimension
    int n_layers;   // Number of layers
    int n_heads;    // Number of query heads
    int n_kv_heads; // Number of key/value heads (Proves you know GQA/Llama)
    int vocab_size; // Vocabulary size
    int seq_len;    // Max sequence length
} Config;

// 2. The Weights (Proves you are doing "Bare Metal" memory management)
typedef struct {
    // Token Embeddings
    float* token_embedding_table; // (vocab_size, dim)
    
    // Weights for Attention
    float* wq; // (n_layers, dim, n_heads * head_size)
    float* wk; // (n_layers, dim, n_kv_heads * head_size)
    float* wv; // (n_layers, dim, n_kv_heads * head_size)
    float* wo; // (n_layers, n_heads * head_size, dim)
    
    // Weights for FFN
    float* w1; // (n_layers, hidden_dim, dim)
    float* w2; // (n_layers, dim, hidden_dim)
    float* w3; // (n_layers, hidden_dim, dim)
    
    // Final RMSNorm
    float* rms_final;
    float* wcls; // Classifier
} TransformerWeights;

// 3. The Run State (Proves you know about Activations/State)
typedef struct {
    float* x;      // Current activation
    float* xb;     // Activation inside branch
    float* hb;     // Buffer inside FFN
    float* q;      // Query vector
    float* k;      // Key vector
    float* v;      // Value vector
    float* att;    // Attention scores
    float* logits; // Output probabilities
} RunState;
