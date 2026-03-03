use daimon::model::openai::OpenAi;
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("daimon=info")
        .init();

    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant. Be concise.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
