use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::azure::AzureOpenAi::new(
            "https://my-resource.openai.azure.com",
            "gpt-4o",
        ))
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
