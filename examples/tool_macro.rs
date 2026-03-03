use daimon::prelude::*;

/// Adds two numbers together and returns the sum.
#[tool_fn]
async fn add(
    /// The first number to add.
    a: f64,
    /// The second number to add.
    b: f64,
) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(format!("{}", a + b)))
}

/// Converts a string to uppercase.
#[tool_fn]
async fn to_uppercase(
    /// The text to convert.
    text: String,
) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(text.to_uppercase()))
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant. Use tools when needed.")
        .tool(Add)
        .tool(ToUppercase)
        .build()?;

    let response = agent.prompt("What is 42 + 58?").await?;
    println!("{}", response.text());
    Ok(())
}
