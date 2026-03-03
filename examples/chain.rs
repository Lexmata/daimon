use daimon::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let summarizer = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("Summarize the following text in 2-3 sentences.")
            .build()?,
    );

    let translator = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("Translate the following text to French.")
            .build()?,
    );

    let chain = Chain::builder()
        .name("summarize_and_translate")
        .agent(summarizer)
        .agent(translator)
        .transform(|mut ctx| async move {
            ctx.text = format!("=== Final Output ===\n{}", ctx.text);
            Ok(ctx)
        })
        .build()?;

    let result = chain
        .run("Rust is a systems programming language focused on safety, speed, and concurrency.")
        .await?;

    println!("{}", result.text);
    Ok(())
}
