fn main() {
    if std::env::var("CARGO_FEATURE_GRPC").is_ok() {
        let mut protos = vec!["proto/daimon_distributed.proto"];

        if std::env::var("CARGO_FEATURE_MCP").is_ok() {
            protos.push("proto/daimon_mcp.proto");
        }

        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&protos, &["proto"])
            .expect("failed to compile proto files");
    }
}
