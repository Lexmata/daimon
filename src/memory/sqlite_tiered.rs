//! SQLite-backed core, archival, and episodic memory.
//!
//! Requires the `sqlite` feature flag. Follows the same connection/session
//! pattern as [`SqliteMemory`](super::SqliteMemory): a `tokio::sync::Mutex`-guarded
//! `rusqlite::Connection`, all operations run via `spawn_blocking`.

use std::collections::HashMap;
use std::sync::Arc;

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};
use crate::memory::{ArchivalMemory, ArchivalRecord, CoreMemory, CoreMemoryBlock};
use crate::memory::{EpisodicEvent, EpisodicMemory, EpisodicQuery};

async fn spawn_blocking_err<T: Send + 'static>(
    f: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))?
}

// ---------------------------------------------------------------------
// Core memory
// ---------------------------------------------------------------------

/// SQLite-backed [`CoreMemory`]. Persists blocks keyed by `(session_id, label)`.
pub struct SqliteCoreMemory {
    conn: Arc<Mutex<Connection>>,
    session_id: String,
}

impl SqliteCoreMemory {
    /// Opens (or creates) a SQLite database at the given path.
    pub async fn open(path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        let conn = tokio::task::spawn_blocking(move || {
            Connection::open(&path).map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))??;
        Self::from_connection(conn, default_session_id()).await
    }

    /// Creates an in-memory SQLite database (data lost when dropped).
    pub async fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))?;
        Self::from_connection(conn, default_session_id()).await
    }

    /// Sets a custom session ID for partitioning core memory blocks.
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    async fn from_connection(conn: Connection, session_id: String) -> Result<Self> {
        let conn = Arc::new(Mutex::new(conn));
        let mem = Self { conn, session_id };
        mem.create_tables().await?;
        Ok(mem)
    }

    async fn create_tables(&self) -> Result<()> {
        let conn = self.conn.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS core_memory_blocks (
                    session_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    value TEXT NOT NULL,
                    char_limit INTEGER,
                    PRIMARY KEY (session_id, label)
                );",
            )
            .map_err(|e| DaimonError::Other(format!("sqlite create tables: {e}")))
        })
        .await
    }
}

fn check_limit(label: &str, value: &str, limit: Option<i64>) -> Result<()> {
    if let Some(limit) = limit
        && value.chars().count() as i64 > limit
    {
        return Err(DaimonError::Other(format!(
            "core memory block '{label}' exceeds limit of {limit} characters"
        )));
    }
    Ok(())
}

impl CoreMemory for SqliteCoreMemory {
    async fn blocks(&self) -> Result<Vec<CoreMemoryBlock>> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT label, value, char_limit FROM core_memory_blocks
                     WHERE session_id = ?1 ORDER BY rowid ASC",
                )
                .map_err(|e| DaimonError::Other(format!("sqlite prepare: {e}")))?;
            let rows = stmt
                .query_map(params![session_id], |row| {
                    Ok(CoreMemoryBlock {
                        label: row.get(0)?,
                        value: row.get(1)?,
                        limit: row.get::<_, Option<i64>>(2)?.map(|n| n as usize),
                    })
                })
                .map_err(|e| DaimonError::Other(format!("sqlite query: {e}")))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| DaimonError::Other(format!("sqlite row: {e}")))
        })
        .await
    }

    async fn put_block(&self, block: CoreMemoryBlock) -> Result<()> {
        check_limit(&block.label, &block.value, block.limit.map(|l| l as i64))?;
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT INTO core_memory_blocks (session_id, label, value, char_limit)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(session_id, label) DO UPDATE SET value = excluded.value, char_limit = excluded.char_limit",
                params![session_id, block.label, block.value, block.limit.map(|l| l as i64)],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite upsert: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn append_block(&self, label: &str, text: &str) -> Result<()> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        let label = label.to_string();
        let text = text.to_string();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            let existing: Option<(String, Option<i64>)> = conn
                .query_row(
                    "SELECT value, char_limit FROM core_memory_blocks WHERE session_id = ?1 AND label = ?2",
                    params![session_id, label],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(|e| DaimonError::Other(format!("sqlite query: {e}")))?;

            let (new_value, limit) = match existing {
                Some((value, limit)) => (format!("{value}{text}"), limit),
                None => (text.clone(), None),
            };
            check_limit(&label, &new_value, limit)?;

            conn.execute(
                "INSERT INTO core_memory_blocks (session_id, label, value, char_limit)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(session_id, label) DO UPDATE SET value = excluded.value",
                params![session_id, label, new_value, limit],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite upsert: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn remove_block(&self, label: &str) -> Result<bool> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        let label = label.to_string();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            let changed = conn
                .execute(
                    "DELETE FROM core_memory_blocks WHERE session_id = ?1 AND label = ?2",
                    params![session_id, label],
                )
                .map_err(|e| DaimonError::Other(format!("sqlite delete: {e}")))?;
            Ok(changed > 0)
        })
        .await
    }
}

// ---------------------------------------------------------------------
// Archival memory (FTS5)
// ---------------------------------------------------------------------

/// SQLite FTS5-backed [`ArchivalMemory`] for consumers without a vector
/// store configured. Metadata is stored as a JSON blob.
pub struct SqliteArchivalMemory {
    conn: Arc<Mutex<Connection>>,
    session_id: String,
}

impl SqliteArchivalMemory {
    /// Opens (or creates) a SQLite database at the given path.
    pub async fn open(path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        let conn = tokio::task::spawn_blocking(move || {
            Connection::open(&path).map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))??;
        Self::from_connection(conn, default_session_id()).await
    }

    /// Creates an in-memory SQLite database (data lost when dropped).
    pub async fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))?;
        Self::from_connection(conn, default_session_id()).await
    }

    /// Sets a custom session ID for partitioning archival facts.
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    async fn from_connection(conn: Connection, session_id: String) -> Result<Self> {
        let conn = Arc::new(Mutex::new(conn));
        let mem = Self { conn, session_id };
        mem.create_tables().await?;
        Ok(mem)
    }

    async fn create_tables(&self) -> Result<()> {
        let conn = self.conn.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            // `id` is derived from `seq` — an `INTEGER PRIMARY KEY
            // AUTOINCREMENT` column — at insert time (see `insert()`), not
            // from an in-process counter. Plain SQLite rowids are just
            // `max(existing rowid) + 1`, recomputed from current table
            // contents; deleting the row holding the current max rowid
            // makes the very next insert reuse that same rowid (and thus
            // the same derived id string). `AUTOINCREMENT` instead tracks a
            // persisted high-water mark in `sqlite_sequence` that never
            // goes backwards, even across deletes, so ids can't collide
            // after a delete+reinsert. It's still true that SQLite
            // allocates rowids atomically per connection under its own
            // locking, so two `SqliteArchivalMemory` instances writing to
            // the same file concurrently can never mint the same id either.
            // The UNIQUE index is defense in depth on top of both of those.
            //
            // NOTE: `CREATE TABLE IF NOT EXISTS` means this does *not*
            // migrate a database file created before this fix (plain `id
            // TEXT NOT NULL`, no `seq` column) — such files keep the old
            // rowid-reuse exposure and may already contain duplicate ids
            // from it. There is no migration path here; a pre-existing file
            // in that state needs manual inspection/deduplication (see the
            // diagnostic on the UNIQUE index creation below) before reuse.
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS archival_facts (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS archival_facts_fts USING fts5(
                    id UNINDEXED, session_id UNINDEXED, text
                );",
            )
            .map_err(|e| DaimonError::Other(format!("sqlite create tables: {e}")))?;

            // Split out from the batch above so a failure here (most likely:
            // a pre-existing database file that already has duplicate `id`
            // values from the old rowid-reuse bug) gets a clear diagnostic
            // instead of an opaque SQLite constraint error.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_archival_facts_id ON archival_facts(id)",
                [],
            )
            .map_err(|e| {
                DaimonError::Other(format!(
                    "sqlite create unique index on archival_facts(id): {e}. This usually means \
                     this database file predates the AUTOINCREMENT id fix (DAIM-32) and already \
                     contains duplicate ids from the earlier rowid-reuse bug. Inspect with \
                     `SELECT id, COUNT(*) FROM archival_facts GROUP BY id HAVING COUNT(*) > 1` \
                     and deduplicate manually before reusing this file."
                ))
            })?;
            Ok(())
        })
        .await
    }
}

/// Inserts a placeholder row, derives the real id from the table's
/// `INTEGER PRIMARY KEY AUTOINCREMENT` sequence value (via
/// `tx.last_insert_rowid()`, which aliases that column), and backfills the
/// `id` column with it in the same transaction. Shared by
/// `SqliteArchivalMemory::insert` and `SqliteEpisodicMemory::record`, which
/// otherwise duplicated this exact sequence.
fn insert_with_sequence_id(
    tx: &rusqlite::Transaction<'_>,
    table: &str,
    id_prefix: &str,
    insert_sql: &str,
    insert_params: &[&dyn rusqlite::ToSql],
) -> Result<String> {
    tx.execute(insert_sql, insert_params)
        .map_err(|e| DaimonError::Other(format!("sqlite insert: {e}")))?;
    let seq = tx.last_insert_rowid();
    let id = format!("{id_prefix}-{seq}");
    tx.execute(
        &format!("UPDATE {table} SET id = ?1 WHERE rowid = ?2"),
        params![id, seq],
    )
    .map_err(|e| DaimonError::Other(format!("sqlite id backfill: {e}")))?;
    Ok(id)
}

impl ArchivalMemory for SqliteArchivalMemory {
    async fn insert(&self, text: &str, metadata: HashMap<String, Value>) -> Result<String> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        let text = text.to_string();
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| DaimonError::Other(format!("serialize metadata: {e}")))?;
        spawn_blocking_err(move || {
            let mut conn = conn.blocking_lock();
            let tx = conn
                .transaction()
                .map_err(|e| DaimonError::Other(format!("sqlite tx begin: {e}")))?;
            // Insert with a placeholder id, then derive the real id from the
            // table's AUTOINCREMENT sequence value (see `create_tables` for
            // why that — not a plain rowid — is required) and back-fill it
            // in the same transaction before the FTS row is written.
            let id = insert_with_sequence_id(
                &tx,
                "archival_facts",
                "archival",
                "INSERT INTO archival_facts (id, session_id, text, metadata) VALUES ('', ?1, ?2, ?3)",
                params![session_id, text, metadata_json],
            )?;
            tx.execute(
                "INSERT INTO archival_facts_fts (id, session_id, text) VALUES (?1, ?2, ?3)",
                params![id, session_id, text],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite fts insert: {e}")))?;
            tx.commit()
                .map_err(|e| DaimonError::Other(format!("sqlite tx commit: {e}")))?;
            Ok(id)
        })
        .await
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ArchivalRecord>> {
        if query.trim().is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        // FTS5 query syntax treats bare terms as an implicit AND with
        // special characters; quote each whitespace-split term so raw user
        // text (punctuation, hyphens) doesn't break the query syntax.
        let fts_query = query
            .split_whitespace()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");

        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT f.id, f.text, f.metadata, bm25(archival_facts_fts) AS rank
                     FROM archival_facts_fts fts
                     JOIN archival_facts f ON f.id = fts.id
                     WHERE archival_facts_fts MATCH ?1 AND fts.session_id = ?2
                     ORDER BY rank LIMIT ?3",
                )
                .map_err(|e| DaimonError::Other(format!("sqlite prepare: {e}")))?;

            let rows = stmt
                .query_map(params![fts_query, session_id, top_k as i64], |row| {
                    let id: String = row.get(0)?;
                    let text: String = row.get(1)?;
                    let metadata_json: String = row.get(2)?;
                    // bm25() scores are negative and lower-is-better; expose
                    // the negated value so higher-is-more-relevant holds
                    // across all ArchivalMemory implementations.
                    let rank: f64 = row.get(3)?;
                    Ok((id, text, metadata_json, -rank))
                })
                .map_err(|e| DaimonError::Other(format!("sqlite query: {e}")))?;

            let mut results = Vec::new();
            for row in rows {
                let (id, text, metadata_json, score) =
                    row.map_err(|e| DaimonError::Other(format!("sqlite row: {e}")))?;
                let metadata: HashMap<String, Value> = serde_json::from_str(&metadata_json)
                    .map_err(|e| DaimonError::Other(format!("corrupted metadata JSON: {e}")))?;
                results.push(ArchivalRecord {
                    id,
                    text,
                    metadata,
                    score: Some(score),
                });
            }
            Ok(results)
        })
        .await
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let conn = self.conn.clone();
        let id = id.to_string();
        let session_id = self.session_id.clone();
        spawn_blocking_err(move || {
            let mut conn = conn.blocking_lock();
            let tx = conn
                .transaction()
                .map_err(|e| DaimonError::Other(format!("sqlite tx begin: {e}")))?;
            let changed = tx
                .execute(
                    "DELETE FROM archival_facts WHERE id = ?1 AND session_id = ?2",
                    params![id, session_id],
                )
                .map_err(|e| DaimonError::Other(format!("sqlite delete: {e}")))?;
            tx.execute(
                "DELETE FROM archival_facts_fts WHERE id = ?1 AND session_id = ?2",
                params![id, session_id],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite fts delete: {e}")))?;
            tx.commit()
                .map_err(|e| DaimonError::Other(format!("sqlite tx commit: {e}")))?;
            Ok(changed > 0)
        })
        .await
    }

    async fn count(&self) -> Result<usize> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM archival_facts WHERE session_id = ?1",
                    params![session_id],
                    |row| row.get(0),
                )
                .map_err(|e| DaimonError::Other(format!("sqlite count: {e}")))?;
            Ok(count as usize)
        })
        .await
    }
}

// ---------------------------------------------------------------------
// Episodic memory
// ---------------------------------------------------------------------

/// SQLite-backed [`EpisodicMemory`] event log.
pub struct SqliteEpisodicMemory {
    conn: Arc<Mutex<Connection>>,
    session_id: String,
}

impl SqliteEpisodicMemory {
    /// Opens (or creates) a SQLite database at the given path.
    pub async fn open(path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        let conn = tokio::task::spawn_blocking(move || {
            Connection::open(&path).map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))??;
        Self::from_connection(conn, default_session_id()).await
    }

    /// Creates an in-memory SQLite database (data lost when dropped).
    pub async fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))?;
        Self::from_connection(conn, default_session_id()).await
    }

    /// Sets a custom session ID for partitioning episodic events.
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    async fn from_connection(conn: Connection, session_id: String) -> Result<Self> {
        let conn = Arc::new(Mutex::new(conn));
        let mem = Self { conn, session_id };
        mem.create_tables().await?;
        Ok(mem)
    }

    async fn create_tables(&self) -> Result<()> {
        let conn = self.conn.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            // Like `archival_facts`, `id` is derived from `seq` — an
            // `INTEGER PRIMARY KEY AUTOINCREMENT` column — at insert time
            // rather than an in-process counter or a plain rowid — see the
            // comment on `SqliteArchivalMemory::create_tables` for why a
            // plain rowid isn't enough (it gets reused after a delete of the
            // current max-rowid row). The same "no migration for pre-existing
            // files" caveat applies here too.
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS episodic_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodic_session_ts
                    ON episodic_events(session_id, timestamp_ms);",
            )
            .map_err(|e| DaimonError::Other(format!("sqlite create tables: {e}")))?;

            // Split out so a failure (most likely a pre-existing database
            // file that already has duplicate `id` values from the old
            // rowid-reuse bug) gets a clear diagnostic instead of an opaque
            // SQLite constraint error.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodic_events_id ON episodic_events(id)",
                [],
            )
            .map_err(|e| {
                DaimonError::Other(format!(
                    "sqlite create unique index on episodic_events(id): {e}. This usually means \
                     this database file predates the AUTOINCREMENT id fix (DAIM-32) and already \
                     contains duplicate ids from the earlier rowid-reuse bug. Inspect with \
                     `SELECT id, COUNT(*) FROM episodic_events GROUP BY id HAVING COUNT(*) > 1` \
                     and deduplicate manually before reusing this file."
                ))
            })?;
            Ok(())
        })
        .await
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

impl EpisodicMemory for SqliteEpisodicMemory {
    async fn record(&self, event_type: &str, payload: Value) -> Result<String> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        let event_type = event_type.to_string();
        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| DaimonError::Other(format!("serialize payload: {e}")))?;
        let ts = now_ms();
        spawn_blocking_err(move || {
            let mut conn = conn.blocking_lock();
            let tx = conn
                .transaction()
                .map_err(|e| DaimonError::Other(format!("sqlite tx begin: {e}")))?;
            // Same sequence-derived id scheme as `SqliteArchivalMemory::insert`.
            let id = insert_with_sequence_id(
                &tx,
                "episodic_events",
                "event",
                "INSERT INTO episodic_events (id, session_id, event_type, payload, timestamp_ms)
                 VALUES ('', ?1, ?2, ?3, ?4)",
                params![session_id, event_type, payload_json, ts],
            )?;
            tx.commit()
                .map_err(|e| DaimonError::Other(format!("sqlite tx commit: {e}")))?;
            Ok(id)
        })
        .await
    }

    async fn query(&self, query: EpisodicQuery) -> Result<Vec<EpisodicEvent>> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            let mut sql = String::from(
                "SELECT id, event_type, payload, timestamp_ms FROM episodic_events WHERE session_id = ?1",
            );
            let mut bind_params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(session_id)];

            if let Some(event_type) = &query.event_type {
                sql.push_str(&format!(" AND event_type = ?{}", bind_params.len() + 1));
                bind_params.push(Box::new(event_type.clone()));
            }
            if let Some(since) = query.since_ms {
                sql.push_str(&format!(" AND timestamp_ms >= ?{}", bind_params.len() + 1));
                bind_params.push(Box::new(since));
            }
            if let Some(until) = query.until_ms {
                sql.push_str(&format!(" AND timestamp_ms <= ?{}", bind_params.len() + 1));
                bind_params.push(Box::new(until));
            }
            // Break ties on identical millisecond timestamps by insertion
            // order (rowid) so "most recent first" is deterministic even
            // for events recorded in quick succession.
            sql.push_str(" ORDER BY timestamp_ms DESC, rowid DESC");
            if let Some(limit) = query.limit {
                sql.push_str(&format!(" LIMIT {limit}"));
            }

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| DaimonError::Other(format!("sqlite prepare: {e}")))?;
            let param_refs: Vec<&dyn rusqlite::ToSql> =
                bind_params.iter().map(|b| b.as_ref()).collect();

            let rows = stmt
                .query_map(param_refs.as_slice(), |row| {
                    let id: String = row.get(0)?;
                    let event_type: String = row.get(1)?;
                    let payload_json: String = row.get(2)?;
                    let timestamp_ms: i64 = row.get(3)?;
                    Ok((id, event_type, payload_json, timestamp_ms))
                })
                .map_err(|e| DaimonError::Other(format!("sqlite query: {e}")))?;

            let mut events = Vec::new();
            for row in rows {
                let (id, event_type, payload_json, timestamp_ms) =
                    row.map_err(|e| DaimonError::Other(format!("sqlite row: {e}")))?;
                let payload: Value = serde_json::from_str(&payload_json)
                    .map_err(|e| DaimonError::Other(format!("corrupted payload JSON: {e}")))?;
                events.push(EpisodicEvent {
                    id,
                    event_type,
                    payload,
                    timestamp_ms,
                });
            }
            Ok(events)
        })
        .await
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        spawn_blocking_err(move || {
            let conn = conn.blocking_lock();
            conn.execute(
                "DELETE FROM episodic_events WHERE session_id = ?1",
                params![session_id],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite delete: {e}")))?;
            Ok(())
        })
        .await
    }
}

/// Generates a session identifier, matching [`SqliteMemory`](super::SqliteMemory)'s scheme.
fn default_session_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let random = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();

    format!("{ts:032x}-{pid:08x}-{count:016x}-{random:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::EpisodicQuery;

    // --- CoreMemory ---

    #[tokio::test]
    async fn core_put_and_get_block() {
        let mem = SqliteCoreMemory::in_memory().await.unwrap();
        mem.put_block(CoreMemoryBlock::new("persona", "helpful"))
            .await
            .unwrap();
        let blocks = mem.blocks().await.unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].value, "helpful");
    }

    #[tokio::test]
    async fn core_put_block_overwrites() {
        let mem = SqliteCoreMemory::in_memory().await.unwrap();
        mem.put_block(CoreMemoryBlock::new("persona", "v1"))
            .await
            .unwrap();
        mem.put_block(CoreMemoryBlock::new("persona", "v2"))
            .await
            .unwrap();
        let blocks = mem.blocks().await.unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].value, "v2");
    }

    #[tokio::test]
    async fn core_append_respects_limit() {
        let mem = SqliteCoreMemory::in_memory().await.unwrap();
        mem.put_block(CoreMemoryBlock::new("notes", "1234").with_limit(5))
            .await
            .unwrap();
        mem.append_block("notes", "5").await.unwrap();
        assert!(mem.append_block("notes", "6").await.is_err());
        assert_eq!(mem.blocks().await.unwrap()[0].value, "12345");
    }

    #[tokio::test]
    async fn core_remove_block() {
        let mem = SqliteCoreMemory::in_memory().await.unwrap();
        mem.put_block(CoreMemoryBlock::new("a", "x")).await.unwrap();
        assert!(mem.remove_block("a").await.unwrap());
        assert!(!mem.remove_block("a").await.unwrap());
    }

    #[tokio::test]
    async fn core_session_isolation() {
        let mem1 = SqliteCoreMemory::in_memory().await.unwrap();
        let mem2 = SqliteCoreMemory {
            conn: mem1.conn.clone(),
            session_id: "other".into(),
        };
        mem1.put_block(CoreMemoryBlock::new("a", "1"))
            .await
            .unwrap();
        assert!(mem2.blocks().await.unwrap().is_empty());
    }

    // --- ArchivalMemory ---

    #[tokio::test]
    async fn archival_insert_and_search() {
        let mem = SqliteArchivalMemory::in_memory().await.unwrap();
        let id = mem.insert("the sky is blue", HashMap::new()).await.unwrap();
        mem.insert("grass is green", HashMap::new()).await.unwrap();

        let results = mem.search("sky", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[tokio::test]
    async fn archival_search_empty_query_returns_empty() {
        let mem = SqliteArchivalMemory::in_memory().await.unwrap();
        mem.insert("fact", HashMap::new()).await.unwrap();
        assert!(mem.search("   ", 5).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn archival_delete_and_count() {
        let mem = SqliteArchivalMemory::in_memory().await.unwrap();
        let id = mem.insert("fact one", HashMap::new()).await.unwrap();
        mem.insert("fact two", HashMap::new()).await.unwrap();
        assert_eq!(mem.count().await.unwrap(), 2);
        assert!(mem.delete(&id).await.unwrap());
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    /// bm25's raw score is lower-is-better; `search` negates it so
    /// higher-is-more-relevant holds across every `ArchivalMemory`
    /// implementation (see the comment at the negation site). A regression
    /// that dropped or double-applied that negation would silently invert
    /// the ranking below with nothing else catching it.
    #[tokio::test]
    async fn archival_more_relevant_fact_ranks_first() {
        let mem = SqliteArchivalMemory::in_memory().await.unwrap();
        // Insert the weaker match first so a regression to
        // insertion-order-as-tiebreak wouldn't accidentally pass.
        let id_weak = mem
            .insert("the ocean is deep and blue", HashMap::new())
            .await
            .unwrap();
        let id_strong = mem
            .insert("sky blue sky blue sky blue", HashMap::new())
            .await
            .unwrap();

        let results = mem.search("sky blue", 5).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].id, id_strong,
            "the denser term match must rank first"
        );
        assert_eq!(results[1].id, id_weak);
        assert!(results[0].score.unwrap() > results[1].score.unwrap());
    }

    #[tokio::test]
    async fn archival_metadata_round_trips() {
        let mem = SqliteArchivalMemory::in_memory().await.unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String("wiki".into()));
        mem.insert("a fact", metadata).await.unwrap();
        let results = mem.search("fact", 1).await.unwrap();
        assert_eq!(results[0].metadata["source"], Value::String("wiki".into()));
    }

    #[tokio::test]
    async fn archival_query_with_punctuation_does_not_error() {
        let mem = SqliteArchivalMemory::in_memory().await.unwrap();
        mem.insert("cost is $5.00 (approx)", HashMap::new())
            .await
            .unwrap();
        let results = mem.search("$5.00 (approx)?", 5).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn archival_session_isolation() {
        let mem1 = SqliteArchivalMemory::in_memory().await.unwrap();
        let mem2 = SqliteArchivalMemory {
            conn: mem1.conn.clone(),
            session_id: "other".into(),
        };
        let id1 = mem1
            .insert("session one fact", HashMap::new())
            .await
            .unwrap();

        // Session 2 cannot delete session 1's fact by guessing its id.
        assert!(!mem2.delete(&id1).await.unwrap());
        assert_eq!(mem1.count().await.unwrap(), 1);
        let results = mem1.search("session one fact", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id1);
    }

    #[tokio::test]
    async fn archival_restart_produces_distinct_ids() {
        let path = std::env::temp_dir().join(format!(
            "daimon-archival-restart-{}-{}.sqlite",
            std::process::id(),
            now_ms()
        ));
        let path_str = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        let id1 = {
            let mem = SqliteArchivalMemory::open(path_str.clone())
                .await
                .unwrap()
                .with_session_id("fixed-session");
            mem.insert("first fact", HashMap::new()).await.unwrap()
        };

        // Reopen against the same file-backed DB, simulating a restart.
        let mem = SqliteArchivalMemory::open(path_str.clone())
            .await
            .unwrap()
            .with_session_id("fixed-session");
        let id2 = mem.insert("second fact", HashMap::new()).await.unwrap();
        assert_ne!(id1, id2, "restart must not reuse an id already persisted");
        assert_eq!(mem.count().await.unwrap(), 2);

        let _ = std::fs::remove_file(&path_str);
    }

    /// Regression test for the finding in the DAIM-25 review: seeding an
    /// in-process `AtomicU64` from persisted state at construction only
    /// fixed the restart-after-close case. Two `SqliteArchivalMemory`
    /// instances can be legitimately live at once against the same file
    /// (that's what the `session_id` partitioning exists for), and a
    /// per-instance counter has no way to observe inserts the *other* live
    /// instance makes after both have already seeded. Ids must instead come
    /// from the database itself (here: the table's own `rowid`, allocated
    /// atomically by SQLite per-connection), so two concurrently-live
    /// instances can never collide regardless of in-process state.
    #[tokio::test]
    async fn archival_concurrent_instances_get_distinct_ids() {
        let path = std::env::temp_dir().join(format!(
            "daimon-archival-concurrent-{}-{}.sqlite",
            std::process::id(),
            now_ms()
        ));
        let path_str = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        // Two independently-opened instances (separate `Connection`s), both
        // live at once, pointed at the same file and the same session.
        let a = SqliteArchivalMemory::open(path_str.clone())
            .await
            .unwrap()
            .with_session_id("shared-session");
        let b = SqliteArchivalMemory::open(path_str.clone())
            .await
            .unwrap()
            .with_session_id("shared-session");

        // Actually race the two handles' writes against each other — issuing
        // them as sequential `.await`s on one task (as an earlier version of
        // this test did) never lets them overlap, so it can't catch a real
        // concurrency bug. `tokio::join!` drives both futures concurrently;
        // repeat it a few rounds to raise the odds of genuine overlap.
        let mut all_ids = Vec::new();
        for round in 0..5 {
            let text_a = format!("fact from a round {round} alpha-marker");
            let text_b = format!("fact from b round {round} beta-marker");
            let (res_a, res_b) = tokio::join!(
                a.insert(&text_a, HashMap::new()),
                b.insert(&text_b, HashMap::new()),
            );
            all_ids.push(
                res_a.expect("concurrent insert from instance a must not error (e.g. SQLITE_BUSY)"),
            );
            all_ids.push(
                res_b.expect("concurrent insert from instance b must not error (e.g. SQLITE_BUSY)"),
            );
        }

        let unique: std::collections::HashSet<_> = all_ids.iter().collect();
        assert_eq!(
            unique.len(),
            all_ids.len(),
            "all ids must be distinct: {all_ids:?}"
        );

        // Both instances see all ten facts (same session, same file) — no
        // insert was rejected or silently lost to a PRIMARY KEY collision.
        assert_eq!(a.count().await.unwrap(), 10);
        assert_eq!(b.count().await.unwrap(), 10);

        // The FTS mirror must stay correctly paired with the base table
        // under concurrent writes too — `search()` joins through
        // `archival_facts_fts`, so a mispaired row here would mean the FTS
        // insert and the id backfill raced each other across instances.
        let alpha_hits = a.search("alpha-marker", 10).await.unwrap();
        assert_eq!(
            alpha_hits.len(),
            5,
            "every concurrent insert from instance a must be findable via FTS search"
        );
        let beta_hits = b.search("beta-marker", 10).await.unwrap();
        assert_eq!(
            beta_hits.len(),
            5,
            "every concurrent insert from instance b must be findable via FTS search"
        );

        let _ = std::fs::remove_file(&path_str);
    }

    // --- EpisodicMemory ---

    #[tokio::test]
    async fn episodic_record_and_query() {
        let mem = SqliteEpisodicMemory::in_memory().await.unwrap();
        mem.record("login", serde_json::json!({"user": "a"}))
            .await
            .unwrap();
        mem.record("logout", serde_json::json!({"user": "a"}))
            .await
            .unwrap();

        let all = mem.query(EpisodicQuery::all()).await.unwrap();
        assert_eq!(all.len(), 2);

        let logins = mem
            .query(EpisodicQuery::all().of_type("login"))
            .await
            .unwrap();
        assert_eq!(logins.len(), 1);
        assert_eq!(logins[0].payload["user"], "a");
    }

    #[tokio::test]
    async fn episodic_query_respects_limit_and_recency() {
        let mem = SqliteEpisodicMemory::in_memory().await.unwrap();
        for i in 0..5 {
            mem.record("tick", serde_json::json!({"i": i}))
                .await
                .unwrap();
        }
        let recent = mem.query(EpisodicQuery::all().limit(2)).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].payload["i"], 4);
    }

    #[tokio::test]
    async fn episodic_clear_removes_events() {
        let mem = SqliteEpisodicMemory::in_memory().await.unwrap();
        mem.record("tick", Value::Null).await.unwrap();
        mem.clear().await.unwrap();
        assert!(mem.query(EpisodicQuery::all()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn episodic_restart_produces_distinct_ids() {
        let path = std::env::temp_dir().join(format!(
            "daimon-episodic-restart-{}-{}.sqlite",
            std::process::id(),
            now_ms()
        ));
        let path_str = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        let id1 = {
            let mem = SqliteEpisodicMemory::open(path_str.clone())
                .await
                .unwrap()
                .with_session_id("fixed-session");
            mem.record("tick", Value::Null).await.unwrap()
        };

        // Reopen against the same file-backed DB, simulating a restart.
        let mem = SqliteEpisodicMemory::open(path_str.clone())
            .await
            .unwrap()
            .with_session_id("fixed-session");
        let id2 = mem.record("tick", Value::Null).await.unwrap();
        assert_ne!(id1, id2, "restart must not reuse an id already persisted");
        assert_eq!(mem.query(EpisodicQuery::all()).await.unwrap().len(), 2);

        let _ = std::fs::remove_file(&path_str);
    }

    /// Regression test for the finding in the DAIM-25 review — see
    /// `archival_concurrent_instances_get_distinct_ids` for the full
    /// rationale. Same root cause, same fix, applied to episodic events.
    #[tokio::test]
    async fn episodic_concurrent_instances_get_distinct_ids() {
        let path = std::env::temp_dir().join(format!(
            "daimon-episodic-concurrent-{}-{}.sqlite",
            std::process::id(),
            now_ms()
        ));
        let path_str = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        let a = SqliteEpisodicMemory::open(path_str.clone())
            .await
            .unwrap()
            .with_session_id("shared-session");
        let b = SqliteEpisodicMemory::open(path_str.clone())
            .await
            .unwrap()
            .with_session_id("shared-session");

        // Actually race the two handles' writes against each other, the same
        // way `archival_concurrent_instances_get_distinct_ids` does —
        // sequential `.await`s on one task never let the two connections'
        // writes genuinely overlap.
        let mut all_ids = Vec::new();
        for _ in 0..5 {
            let (res_a, res_b) =
                tokio::join!(a.record("tick", Value::Null), b.record("tick", Value::Null),);
            all_ids.push(
                res_a.expect("concurrent record from instance a must not error (e.g. SQLITE_BUSY)"),
            );
            all_ids.push(
                res_b.expect("concurrent record from instance b must not error (e.g. SQLITE_BUSY)"),
            );
        }

        let unique: std::collections::HashSet<_> = all_ids.iter().collect();
        assert_eq!(
            unique.len(),
            all_ids.len(),
            "all ids must be distinct: {all_ids:?}"
        );
        assert_eq!(
            a.query(EpisodicQuery::all()).await.unwrap().len(),
            10,
            "no insert should have been rejected or lost to an id collision"
        );

        let _ = std::fs::remove_file(&path_str);
    }

    #[tokio::test]
    async fn episodic_session_isolation() {
        let mem1 = SqliteEpisodicMemory::in_memory().await.unwrap();
        let mem2 = SqliteEpisodicMemory {
            conn: mem1.conn.clone(),
            session_id: "other".into(),
        };
        mem1.record("tick", Value::Null).await.unwrap();
        assert!(mem2.query(EpisodicQuery::all()).await.unwrap().is_empty());
    }
}
