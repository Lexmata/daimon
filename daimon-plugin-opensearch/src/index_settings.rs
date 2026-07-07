//! Index creation JSON for manual setup.
//!
//! When [`OpenSearchVectorStoreBuilder::auto_create_index(false)`](crate::OpenSearchVectorStoreBuilder::auto_create_index)
//! is set, use these helpers to create the index yourself via the OpenSearch API.

use serde_json::{Value, json};

use crate::{Engine, SpaceType};

/// Returns the index creation body as a [`serde_json::Value`].
///
/// The body includes:
/// - `settings.index.knn: true`
/// - A `knn_vector` mapping for the `embedding` field
/// - HNSW method parameters (engine, space type, m, ef_construction)
/// - `content` (text) and `metadata` (object) fields
///
/// # Example
///
/// ```
/// use daimon_plugin_opensearch::index_settings;
/// use daimon_plugin_opensearch::{Engine, SpaceType};
///
/// let body = index_settings::create_index_body(
///     1536,
///     SpaceType::CosineSimilarity,
///     Engine::Lucene,
///     None,
///     None,
/// );
/// assert!(body["settings"]["index"]["knn"].as_bool().unwrap());
/// ```
pub fn create_index_body(
    dimensions: usize,
    space_type: SpaceType,
    engine: Engine,
    m: Option<usize>,
    ef_construction: Option<usize>,
) -> Value {
    let mut params = serde_json::Map::new();
    if let Some(m) = m {
        params.insert("m".into(), json!(m));
    }
    if let Some(ef) = ef_construction {
        params.insert("ef_construction".into(), json!(ef));
    }

    json!({
        "settings": {
            "index": {
                "knn": true
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": space_type.as_str(),
                        "engine": engine.as_str(),
                        "parameters": params
                    }
                },
                "content": {
                    "type": "text"
                },
                "metadata": {
                    "type": "object",
                    "enabled": true
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_index_body_defaults() {
        let body = create_index_body(
            1536,
            SpaceType::CosineSimilarity,
            Engine::Lucene,
            None,
            None,
        );
        assert!(body["settings"]["index"]["knn"].as_bool().unwrap());
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["dimension"],
            1536
        );
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["space_type"],
            "cosinesimil"
        );
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["engine"],
            "lucene"
        );
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["name"],
            "hnsw"
        );
        assert!(
            body["mappings"]["properties"]["embedding"]["method"]["parameters"]
                .as_object()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_create_index_body_custom_params() {
        let body = create_index_body(768, SpaceType::L2, Engine::Faiss, Some(32), Some(256));
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["parameters"]["m"],
            32
        );
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["parameters"]["ef_construction"],
            256
        );
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["space_type"],
            "l2"
        );
        assert_eq!(
            body["mappings"]["properties"]["embedding"]["method"]["engine"],
            "faiss"
        );
    }
}
