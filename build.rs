fn main() {
    if std::env::var("CARGO_FEATURE_GRPC").is_ok() {
        let protos = vec!["proto/daimon_distributed.proto"];

        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&protos, &["proto"])
            .expect("failed to compile proto files");
    }
}
