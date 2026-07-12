use std::sync::atomic::{AtomicUsize, Ordering};

use daimon::error::Result;
use daimon::model::Model;
use daimon::model::types::*;
use daimon::stream::ResponseStream;

pub struct MockModel {
    responses: Vec<ChatResponse>,
    call_count: AtomicUsize,
}

impl MockModel {
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }

    #[allow(dead_code)]
    pub fn single_text(text: &str) -> Self {
        Self::new(vec![ChatResponse {
            message: Message::assistant(text),
            stop_reason: StopReason::EndTurn,
            usage: Some(Usage::default()),
        }])
    }
}

impl Model for MockModel {
    async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        if idx < self.responses.len() {
            Ok(self.responses[idx].clone())
        } else {
            Ok(self.responses.last().unwrap().clone())
        }
    }

    async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}
