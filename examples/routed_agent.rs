//! Dynamic model routing: each ReAct iteration is scored for difficulty and
//! served by the cheapest competent registered model.
//!
//! Set `OPENAI_API_KEY` and run with:
//!   cargo run --example routed_agent --features openai

use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let router = ModelRouter::builder()
        .register(
            ModelTier::Small,
            daimon::model::openai::OpenAi::new("gpt-4o-mini"),
        )
        .register(
            ModelTier::Medium,
            daimon::model::openai::OpenAi::new("gpt-4o"),
        )
        .cost_model(OpenAiCostModel)
        .build()?;

    let agent = Agent::builder().router(router).build()?;

    for prompt in [
        "hi!",
        "Analyze the time-complexity trade-offs of three sorting strategies step by step.",
    ] {
        let response = agent.prompt(prompt).await?;
        println!("Q: {prompt}\nA: {}\n", response.text());
        for d in &response.route_decisions {
            println!(
                "  → difficulty {:.2} routed to {} (tier {:?})",
                d.difficulty, d.selected_model_id, d.selected_tier
            );
        }
    }

    Ok(())
}
