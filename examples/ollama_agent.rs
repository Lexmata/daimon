use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::ollama::Ollama::new("llama3.1"))
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
