use daimon::model::openai::OpenAi;
use daimon::prelude::*;

struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate simple math expressions. Supports add, subtract, multiply, divide."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn execute(&self, input: &Value) -> daimon::Result<ToolOutput> {
        let op = input["operation"].as_str().unwrap_or("add");
        let a = input["a"].as_f64().unwrap_or(0.0);
        let b = input["b"].as_f64().unwrap_or(0.0);

        let result = match op {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Ok(ToolOutput::error("Division by zero"));
                }
                a / b
            }
            _ => return Ok(ToolOutput::error(format!("Unknown operation: {op}"))),
        };

        Ok(ToolOutput::text(format!("{result}")))
    }
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("daimon=info")
        .init();

    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a math tutor. Use the calculator tool to solve problems.")
        .tool(Calculator)
        .max_iterations(10)
        .build()?;

    let response = agent.prompt("What is 42 * 17 + 3?").await?;
    println!("{}", response.text());
    println!("(completed in {} iteration(s))", response.iterations);
    Ok(())
}
