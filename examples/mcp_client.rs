use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let transport = daimon::mcp::StdioTransport::new(
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    .await?;

    let client = daimon::mcp::McpClient::connect(transport).await?;

    println!("Discovered {} MCP tools:", client.tool_infos().len());
    for tool in client.tool_infos() {
        println!(
            "  - {}: {}",
            tool.name,
            tool.description.as_deref().unwrap_or("")
        );
    }

    let mut builder = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant with filesystem access.");

    for tool in client.tools() {
        builder = builder.tool(tool);
    }

    let agent = builder.build()?;
    let response = agent.prompt("List the files in /tmp").await?;
    println!("{}", response.text());

    client.close().await?;
    Ok(())
}
