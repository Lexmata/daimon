//! Amazon Bedrock embedding provider using Amazon Titan Embeddings.
//!
//! Uses the `InvokeModel` API targeting Titan Embedding models.

use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_sdk_bedrockruntime::primitives::Blob;

use daimon_core::{DaimonError, EmbeddingModel, Result};

/// Amazon Bedrock embedding model (Titan Embeddings).
///
/// ```ignore
/// use daimon_provider_bedrock::BedrockEmbedding;
///
/// let embedding = BedrockEmbedding::new("amazon.titan-embed-text-v2:0")
///     .with_region("us-east-1");
/// let vectors = embedding.embed(&["hello world"]).await?;
/// ```
pub struct BedrockEmbedding {
    model_id: String,
    client: Option<BedrockClient>,
    region: Option<String>,
    dimensions: usize,
    normalize: bool,
}

impl BedrockEmbedding {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            client: None,
            region: None,
            dimensions: 1024,
            normalize: true,
        }
    }

    pub fn with_client(mut self, client: BedrockClient) -> Self {
        self.client = Some(client);
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    async fn get_client(&self) -> Result<BedrockClient> {
        if let Some(ref client) = self.client {
            return Ok(client.clone());
        }
        let mut config_loader = aws_config::from_env();
        if let Some(ref region) = self.region {
            config_loader = config_loader.region(aws_config::Region::new(region.clone()));
        }
        let config = config_loader.load().await;
        Ok(BedrockClient::new(&config))
    }
}

impl EmbeddingModel for BedrockEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let client = self.get_client().await?;
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let body = serde_json::json!({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": self.normalize,
            });

            let resp = client
                .invoke_model()
                .model_id(&self.model_id)
                .body(Blob::new(serde_json::to_vec(&body).map_err(|e| {
                    DaimonError::Model(format!("Bedrock embedding serialize error: {e}"))
                })?))
                .content_type("application/json")
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Bedrock embedding error: {e}")))?;

            let output_bytes = resp.body().as_ref();
            let parsed: serde_json::Value =
                serde_json::from_slice(output_bytes).map_err(|e| {
                    DaimonError::Model(format!("Bedrock embedding parse error: {e}"))
                })?;

            let embedding = parsed
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    DaimonError::Model("missing 'embedding' field in Bedrock response".into())
                })?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            results.push(embedding);
        }

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedrock_embedding_new() {
        let embed = BedrockEmbedding::new("amazon.titan-embed-text-v2:0");
        assert_eq!(embed.model_id, "amazon.titan-embed-text-v2:0");
        assert_eq!(embed.dimensions, 1024);
        assert!(embed.normalize);
    }

    #[test]
    fn test_builder_chain() {
        let embed = BedrockEmbedding::new("amazon.titan-embed-text-v2:0")
            .with_region("eu-west-1")
            .with_dimensions(512)
            .with_normalize(false);
        assert_eq!(embed.region.as_deref(), Some("eu-west-1"));
        assert_eq!(embed.dimensions, 512);
        assert!(!embed.normalize);
    }
}
