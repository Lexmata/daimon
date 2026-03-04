//! OpenAI embeddings provider (text-embedding-3-small, text-embedding-3-large, etc.).

use serde::{Deserialize, Serialize};

use crate::error::{DaimonError, Result};
use crate::model::EmbeddingModel;

/// OpenAI embedding model client.
pub struct OpenAiEmbedding {
    client: reqwest::Client,
    api_key: String,
    model_id: String,
    dimensions: usize,
    base_url: String,
}

impl OpenAiEmbedding {
    /// Creates a new OpenAI embedding client.
    ///
    /// Reads `OPENAI_API_KEY` from the environment if `api_key` is not provided.
    pub fn new(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let dimensions = if model_id.contains("large") { 3072 } else { 1536 };
        Self {
            client: reqwest::Client::new(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model_id,
            dimensions,
            base_url: "https://api.openai.com/v1".to_string(),
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
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

impl EmbeddingModel for OpenAiEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedRequest {
            model: &self.model_id,
            input: texts,
        };

        let resp = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("OpenAI embedding HTTP error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "OpenAI embedding error {status}: {text}"
            )));
        }

        let data: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("OpenAI embedding parse error: {e}")))?;

        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
