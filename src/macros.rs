use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(FieldNames)]
pub fn derive_field_names(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = input.ident;

    let field_names = match input.data {
        Data::Struct(data_struct) => {
            match data_struct.fields {
                Fields::Named(fields_named) => {
                    fields_named.named.iter().map(|field| {
                        let field_name = field.ident.as_ref().unwrap().to_string();
                        quote! { #field_name }
                    }).collect::<Vec<_>>()
                }
                _ => panic!("FieldNames can only be derived for structs with named fields"),
            }
        }
        _ => panic!("FieldNames can only be derived for structs"),
    };

    let expanded = quote! {
        impl #struct_name {
            pub const FIELD_NAMES: &'static [&'static str] = &[#(#field_names),*];
        }
    };

    TokenStream::from(expanded)
}