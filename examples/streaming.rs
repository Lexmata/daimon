use daimon::model::openai::OpenAi;
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let mut stream = agent
        .prompt_stream("Explain quantum computing in 3 sentences.")
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => print!("{text}"),
            StreamEvent::ToolCallStart { name, .. } => eprintln!("\n[calling tool: {name}]"),
            StreamEvent::Done => {
                println!();
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
