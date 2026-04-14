#![allow(unused_variables, unused_imports, dead_code, unreachable_patterns)]

pub mod ir;
pub mod lower;
pub mod parser;
#[allow(clippy::not_unsafe_ptr_arg_deref, clippy::needless_range_loop)]
pub mod tensor_rt;
