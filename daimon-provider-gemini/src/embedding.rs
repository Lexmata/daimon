//! Google Gemini embedding model provider.
//!
//! Uses the `embedContent` / `batchEmbedContents` API endpoints.

use serde::{Deserialize, Serialize};

use daimon_core::{DaimonError, EmbeddingModel, Result};

/// Google Gemini embedding model.
///
/// ```ignore
/// use daimon_provider_gemini::GeminiEmbedding;
///
/// let embedding = GeminiEmbedding::new("text-embedding-004");
/// let vectors = embedding.embed(&["hello world"]).await?;
/// ```
pub struct GeminiEmbedding {
    client: reqwest::Client,
    api_key: String,
    model_id: String,
    base_url: String,
    dimensions: usize,
    use_bearer_token: bool,
}

impl GeminiEmbedding {
    /// Creates a new Gemini embedding client, reading `GOOGLE_API_KEY` from env.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
        Self {
            client: reqwest::Client::new(),
            api_key,
            model_id: model_id.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            dimensions: 768,
            use_bearer_token: false,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Use `Authorization: Bearer <key>` instead of `?key=` query parameter.
    pub fn with_bearer_token(mut self) -> Self {
        self.use_bearer_token = true;
        self
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.use_bearer_token {
            req.bearer_auth(&self.api_key)
        } else {
            req.query(&[("key", &self.api_key)])
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchEmbedRequest {
    requests: Vec<EmbedContentRequest>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest {
    model: String,
    content: EmbedContent,
    output_dimensionality: Option<usize>,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<EmbedPart>,
}

#[derive(Serialize)]
struct EmbedPart {
    text: String,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbedValues>,
}

#[derive(Deserialize)]
struct EmbedValues {
    values: Vec<f32>,
}

impl EmbeddingModel for GeminiEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let requests: Vec<EmbedContentRequest> = texts
            .iter()
            .map(|text| EmbedContentRequest {
                model: format!("models/{}", self.model_id),
                content: EmbedContent {
                    parts: vec![EmbedPart {
                        text: text.to_string(),
                    }],
                },
                output_dimensionality: Some(self.dimensions),
            })
            .collect();

        let body = BatchEmbedRequest { requests };
        let url = format!(
            "{}/models/{}:batchEmbedContents",
            self.base_url, self.model_id
        );

        let req = self.client.post(&url).json(&body);
        let req = self.apply_auth(req);

        let resp = req
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("Gemini embedding HTTP error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "Gemini embedding error {status}: {text}"
            )));
        }

        let data: BatchEmbedResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("Gemini embedding parse error: {e}")))?;

        Ok(data.embeddings.into_iter().map(|e| e.values).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_embedding_new() {
        let embed = GeminiEmbedding::new("text-embedding-004");
        assert_eq!(embed.model_id, "text-embedding-004");
        assert_eq!(embed.dimensions, 768);
    }

    #[test]
    fn test_builder_chain() {
        let embed = GeminiEmbedding::new("text-embedding-004")
            .with_api_key("key")
            .with_base_url("https://custom.example.com")
            .with_dimensions(256)
            .with_bearer_token();
        assert_eq!(embed.api_key, "key");
        assert_eq!(embed.base_url, "https://custom.example.com");
        assert_eq!(embed.dimensions, 256);
        assert!(embed.use_bearer_token);
    }
}
