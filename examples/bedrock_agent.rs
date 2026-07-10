use daimon::model::bedrock::Bedrock;
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("daimon=info")
        .init();

    let agent = Agent::builder()
        .model(Bedrock::new("us.anthropic.claude-sonnet-5").with_region("us-east-1"))
        .system_prompt("You are a helpful assistant. Be concise.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
