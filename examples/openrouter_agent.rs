use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(
            daimon::model::openrouter::OpenRouter::new("openai/gpt-4o")
                .with_app_name("daimon-example"),
        )
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
