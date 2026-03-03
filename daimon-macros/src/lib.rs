//! Proc macros for the Daimon AI agent framework.
//!
//! Provides `#[tool_fn]` to derive [`Tool`](https://docs.rs/daimon/latest/daimon/tool/trait.Tool.html)
//! implementations from plain async functions.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Attribute, Expr, FnArg, Ident, ItemFn, Lit, Meta, Pat, PatType, Token, Type,
};

struct ToolFnArgs {
    crate_path: Option<syn::Path>,
    name_override: Option<String>,
    description_override: Option<String>,
}

impl Parse for ToolFnArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut crate_path = None;
        let mut name_override = None;
        let mut description_override = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "crate_path" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Str(s) = lit {
                        crate_path = Some(s.parse()?);
                    }
                }
                "name" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Str(s) = lit {
                        name_override = Some(s.value());
                    }
                }
                "description" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Str(s) = lit {
                        description_override = Some(s.value());
                    }
                }
                other => {
                    return Err(syn::Error::new(key.span(), format!("unknown attribute `{other}`")));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            crate_path,
            name_override,
            description_override,
        })
    }
}

struct ParamInfo {
    name: String,
    ty: Type,
    doc: Option<String>,
    optional: bool,
    inner_ty: Option<Type>,
}

/// Derives a [`Tool`] implementation from an async function.
///
/// The function's parameters become the tool's JSON Schema properties.
/// Doc comments on the function become the tool description; doc comments
/// on individual parameters become property descriptions.
///
/// # Supported types
///
/// `String`, `i8`–`i128`, `u8`–`u128`, `isize`, `usize`, `f32`, `f64`,
/// `bool`, `Option<T>` (marks the parameter as not required).
///
/// # Attributes
///
/// - `name = "..."` — override the tool name (defaults to the function name)
/// - `description = "..."` — override the description (defaults to doc comments)
/// - `crate_path = "..."` — override the path to the daimon crate (defaults to `::daimon`)
///
/// # Example
///
/// ```ignore
/// use daimon::prelude::*;
/// use daimon::tool_fn;
///
/// /// Adds two numbers together.
/// #[tool_fn]
/// async fn add(
///     /// The first number.
///     a: f64,
///     /// The second number.
///     b: f64,
/// ) -> daimon::Result<ToolOutput> {
///     Ok(ToolOutput::text(format!("{}", a + b)))
/// }
///
/// // Use the generated struct:
/// let agent = Agent::builder()
///     .model(model)
///     .tool(Add) // PascalCase struct generated from `add`
///     .build()?;
/// ```
#[proc_macro_attribute]
pub fn tool_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolFnArgs);
    let func = parse_macro_input!(item as ItemFn);

    match expand_tool_fn(args, func) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_tool_fn(args: ToolFnArgs, func: ItemFn) -> syn::Result<TokenStream2> {
    let crate_path = args
        .crate_path
        .map(|p| quote!(#p))
        .unwrap_or_else(|| quote!(::daimon));

    let fn_name = &func.sig.ident;
    let struct_name = format_ident!("{}", to_pascal_case(&fn_name.to_string()));
    let tool_name_str = args.name_override.unwrap_or_else(|| fn_name.to_string());

    let description = args
        .description_override
        .unwrap_or_else(|| extract_doc_comments(&func.attrs));

    if func.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            func.sig.fn_token,
            "tool_fn requires an async function",
        ));
    }

    let params = extract_params(&func)?;
    let schema_tokens = generate_schema(&params, &crate_path);
    let extraction_tokens = generate_extraction(&params, &crate_path);
    let body = &func.block;

    Ok(quote! {
        /// Auto-generated tool struct from `#[tool_fn]` on [`#fn_name`].
        pub struct #struct_name;

        impl #crate_path::tool::Tool for #struct_name {
            fn name(&self) -> &str {
                #tool_name_str
            }

            fn description(&self) -> &str {
                #description
            }

            fn parameters_schema(&self) -> ::serde_json::Value {
                #schema_tokens
            }

            async fn execute(
                &self,
                __daimon_input: &::serde_json::Value,
            ) -> #crate_path::Result<#crate_path::tool::ToolOutput> {
                #extraction_tokens
                #body
            }
        }
    })
}

fn extract_doc_comments(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &attr.meta {
                if let Expr::Lit(lit) = &nv.value {
                    if let Lit::Str(s) = &lit.lit {
                        lines.push(s.value().trim().to_string());
                    }
                }
            }
        }
    }
    lines.join(" ").trim().to_string()
}

fn extract_params(func: &ItemFn) -> syn::Result<Vec<ParamInfo>> {
    let mut params = Vec::new();

    for arg in &func.sig.inputs {
        if let FnArg::Typed(PatType { pat, ty, attrs, .. }) = arg {
            let name = match pat.as_ref() {
                Pat::Ident(ident) => ident.ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(pat, "expected a simple identifier"));
                }
            };

            let doc = extract_doc_comments(attrs);
            let doc = if doc.is_empty() { None } else { Some(doc) };

            let (optional, inner_ty) = unwrap_option(ty);

            params.push(ParamInfo {
                name,
                ty: *ty.clone(),
                doc,
                optional,
                inner_ty,
            });
        }
    }

    Ok(params)
}

fn unwrap_option(ty: &Type) -> (bool, Option<Type>) {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = ab.args.first() {
                        return (true, Some(inner.clone()));
                    }
                }
            }
        }
    }
    (false, None)
}

fn type_to_json_schema(ty: &Type) -> TokenStream2 {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            match name.as_str() {
                "String" | "str" => return quote!(::serde_json::json!({"type": "string"})),
                "bool" => return quote!(::serde_json::json!({"type": "boolean"})),
                "f32" | "f64" => return quote!(::serde_json::json!({"type": "number"})),
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32"
                | "u64" | "u128" | "usize" => {
                    return quote!(::serde_json::json!({"type": "integer"}));
                }
                "Vec" => {
                    if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = ab.args.first() {
                            let inner_schema = type_to_json_schema(inner);
                            return quote!(::serde_json::json!({"type": "array", "items": #inner_schema}));
                        }
                    }
                    return quote!(::serde_json::json!({"type": "array"}));
                }
                "Value" => return quote!(::serde_json::json!({})),
                _ => {}
            }
        }
    }
    quote!(::serde_json::json!({}))
}

fn generate_schema(params: &[ParamInfo], _crate_path: &TokenStream2) -> TokenStream2 {
    let mut prop_entries = Vec::new();
    let mut required_names = Vec::new();

    for param in params {
        let name = &param.name;
        let effective_ty = param.inner_ty.as_ref().unwrap_or(&param.ty);
        let schema = type_to_json_schema(effective_ty);

        if let Some(doc) = &param.doc {
            prop_entries.push(quote! {
                let mut __prop = #schema;
                if let Some(obj) = __prop.as_object_mut() {
                    obj.insert("description".to_string(), ::serde_json::Value::String(#doc.to_string()));
                }
                __props.insert(#name.to_string(), __prop);
            });
        } else {
            prop_entries.push(quote! {
                __props.insert(#name.to_string(), #schema);
            });
        }

        if !param.optional {
            required_names.push(quote!(#name));
        }
    }

    quote! {
        {
            let mut __props = ::serde_json::Map::new();
            #(#prop_entries)*
            let mut __schema = ::serde_json::Map::new();
            __schema.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
            __schema.insert("properties".to_string(), ::serde_json::Value::Object(__props));
            let __required: Vec<&str> = vec![#(#required_names),*];
            if !__required.is_empty() {
                __schema.insert(
                    "required".to_string(),
                    ::serde_json::Value::Array(
                        __required.into_iter().map(|s| ::serde_json::Value::String(s.to_string())).collect()
                    ),
                );
            }
            ::serde_json::Value::Object(__schema)
        }
    }
}

fn generate_extraction(params: &[ParamInfo], crate_path: &TokenStream2) -> TokenStream2 {
    let mut extractions = Vec::new();

    for param in params {
        let name_str = &param.name;
        let name_ident = format_ident!("{}", &param.name);
        let ty = &param.ty;

        if param.optional {
            let inner = param.inner_ty.as_ref().unwrap_or(&param.ty);
            extractions.push(quote! {
                let #name_ident: #ty = match __daimon_input.get(#name_str) {
                    Some(v) if !v.is_null() => {
                        Some(::serde_json::from_value::<#inner>(v.clone()).map_err(|__e| {
                            #crate_path::DaimonError::Other(
                                format!("parameter '{}': {}", #name_str, __e)
                            )
                        })?)
                    }
                    _ => None,
                };
            });
        } else {
            extractions.push(quote! {
                let #name_ident: #ty = ::serde_json::from_value(
                    __daimon_input
                        .get(#name_str)
                        .cloned()
                        .unwrap_or(::serde_json::Value::Null),
                )
                .map_err(|__e| {
                    #crate_path::DaimonError::Other(
                        format!("parameter '{}': {}", #name_str, __e)
                    )
                })?;
            });
        }
    }

    quote! { #(#extractions)* }
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("add"), "Add");
        assert_eq!(to_pascal_case("fetch_weather"), "FetchWeather");
        assert_eq!(to_pascal_case("get_user_by_id"), "GetUserById");
        assert_eq!(to_pascal_case("a"), "A");
    }
}
