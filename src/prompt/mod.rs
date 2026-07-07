//! Composable prompt construction with variable interpolation.
//!
//! [`PromptTemplate`] supports `{variable}` placeholders and composable
//! sections (persona, instructions, examples, constraints). Use
//! [`PromptBuilder`] for fluent construction.
//!
//! ```ignore
//! use daimon::prompt::PromptTemplate;
//!
//! let tpl = PromptTemplate::new("You are {role}. Today is {date}.")
//!     .var("role", "a helpful assistant")
//!     .var("date", "2026-03-03");
//!
//! assert_eq!(tpl.render_static(), "You are a helpful assistant. Today is 2026-03-03.");
//! ```

mod builder;
mod dynamic;
mod few_shot;
mod template;

pub use builder::PromptBuilder;
pub use dynamic::{DynamicContext, ErasedDynamicContext};
pub use few_shot::FewShotTemplate;
pub use template::PromptTemplate;
