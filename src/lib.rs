extern crate libc;
extern crate num_cpus;

// Dummy module just to let `cxx` link against the C++ standard library.
// TODO: What are the magic words to put in `build.rs` to make this unnecessary?
#[cxx::bridge]
mod LLaMaMod {}

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::convert::AsRef;
use std::ffi::CString;
use std::iter::Iterator;
use std::os::raw::{c_char, c_int};
use std::str;
use std::slice;

#[derive(Debug)]
pub struct LLaMaError(&'static str);

pub struct Params {
  pub n_ctx: c_int,
  pub n_parts: c_int,
  pub seed: c_int,
  pub threads: c_int,
  pub embedding: bool,
  pub use_mmap: bool,
  pub use_mlock: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerParams {
  top_p: f32,
  top_k: c_int,
  temperature: f32,
  repeat_penalty: f32,
}

impl SamplerParams {
  pub fn default() -> SamplerParams {
    Self {
      top_p: 0.9,
      top_k: 40,
      repeat_penalty: 1.17,
      temperature: 0.7,
    }
  }
}
  

enum Sampler {
  BuiltinSampler(SamplerParams),
  SamplerFn(fn(&[f32]) -> llama_token)
}


pub struct LLaMaCPP {
  context: *mut llama_context,
  params: Params,
}


impl LLaMaCPP {
  pub fn from_file(path: &str, params: Params) -> Result<LLaMaCPP, LLaMaError> {
    let llama_context_params = llama_context_params {
      n_ctx: params.n_ctx,
      n_parts: params.n_parts,
      seed: params.seed,
      f16_kv: false,
      logits_all: false,
      use_mmap: params.use_mmap,
      use_mlock: params.use_mlock,
      embedding: params.embedding,
      vocab_only: false,
      progress_callback: None,
      progress_callback_user_data: std::ptr::null_mut(),
    };

    let c_path = CString::new(path).map_err(|_| LLaMaError("Invalid path"))?;
    unsafe {
      // TODO: Does this return a null ptr if the file doesn't exist?
      let context = llama_init_from_file(c_path.as_ptr(), llama_context_params);
      if context == std::ptr::null_mut() {
        Err(LLaMaError("Failed to load model"))
      } else {
        Ok(LLaMaCPP { context, params })
      }
    }
  }

  // Lifetime of the returned string is the same as the LLaMa context
  fn token_to_str(&self, token: llama_token) -> &str {
    unsafe {
      let cstr: *const c_char = llama_token_to_str(self.context, token);
      str::from_utf8_unchecked(slice::from_raw_parts(cstr as *const u8, libc::strlen(cstr)))
    }
  }

  fn tokenize_internal(&self, text: &str) -> Vec<llama_token> {
    let c_text = CString::new(text).unwrap();
    let mut tokens = Vec::with_capacity(text.len());
    unsafe {
      let n_tokens = llama_tokenize(self.context, c_text.as_ptr(), tokens.as_mut_ptr(), text.len() as i32, false);
      tokens.set_len(n_tokens as usize);
    }
    tokens
  }

  fn consume_tokens(&mut self, tokens: &[llama_token], n_past: i32) {
    unsafe {
      llama_eval(self.context, tokens.as_ptr(), tokens.len() as i32, n_past, self.params.threads);
    }
  }

  fn sample_token(&self, sampler_params: SamplerParams) -> llama_token {
    unsafe {
      llama_sample_top_p_top_k(self.context, std::ptr::null(), 0, sampler_params.top_k, sampler_params.top_p, sampler_params.temperature, sampler_params.repeat_penalty)
    }
  }

  fn get_last_logits(&self, last_n_tokens: usize) -> &[f32] {
    unsafe {
      let logits_start = llama_get_logits(self.context) as usize;
      slice::from_raw_parts((logits_start + (last_n_tokens - 1) * self.n_vocab() as usize) as *const f32, self.n_vocab() as usize)
    }
  }

  fn get_embeddings(&self) -> Option<&[f32]> {
    if (!self.params.embedding) {
      return None;
    }
    unsafe {
      let embedding = llama_get_embeddings(self.context);
      Some(slice::from_raw_parts(embedding as *const f32, self.n_embd() as usize))
    }
  }

  pub fn n_vocab(&self) -> i32 {
    unsafe { llama_n_vocab(self.context) }
  }

  pub fn n_ctx(&self) -> i32 {
    unsafe { llama_n_ctx(self.context) }
  }

  pub fn n_embd(&self) -> i32 {
    unsafe { llama_n_embd(self.context) }
  }

  pub fn threads(&self) -> i32 {
    self.params.threads
  }

  pub fn system_info() -> &'static str {
    unsafe {
      // This is &'static
      let cstr: *const c_char = llama_print_system_info();
      str::from_utf8_unchecked(slice::from_raw_parts(cstr as *const u8, libc::strlen(cstr)+1))
    }
  }

  pub fn mmap_supported() -> bool {
    unsafe { llama_mmap_supported() }
  }

  pub fn mlock_supported() -> bool {
    unsafe { llama_mlock_supported() }
  }

  fn bos_token() -> llama_token {
    unsafe { llama_token_bos() }
  }

  fn eos_token() -> llama_token {
    unsafe { llama_token_eos() }
  }

  pub fn default_params() -> Params {
    let ffi_params = unsafe { llama_context_default_params() };
    Params {
      n_ctx: ffi_params.n_ctx,
      n_parts: ffi_params.n_parts,
      seed: ffi_params.seed,
      threads: num_cpus::get() as i32,
      embedding: false,
      use_mmap: ffi_params.use_mmap,
      use_mlock: ffi_params.use_mlock,
    }
  }
}

impl Drop for LLaMaCPP {
  fn drop(&mut self) {
    unsafe {
      llama_free(self.context);
    }
  }
}

pub trait LLaMa {
  type Token;
  type TokenIterator<'a>: Iterator where Self: 'a;

  fn tokenize(&self, text: &str) -> Vec<Self::Token>;
  fn to_str(&self, token: Self::Token) -> &str;
  fn bos(&self) -> Self::Token;
  fn eos(&self) -> Self::Token;

  fn iter(&mut self) -> Self::TokenIterator<'_>;
}

pub struct LLaMaCPPTokenIter<'a> {
  context: &'a mut LLaMaCPP,
  sampler: Sampler,
  n_past: i32,
  n_last: i32,
}

impl LLaMaCPPTokenIter<'_> {
  fn consume_internal(&mut self, tokens: &[llama_token]) {
    let new_tokens = tokens.len() as i32;
    let n_ctx = self.context.n_ctx();

    if new_tokens == 0 {
      return;
    } else if new_tokens > n_ctx {
      self.n_last = n_ctx;
      self.context.consume_tokens(&tokens[0..(n_ctx - 1) as usize], 0);
      self.n_past = n_ctx;
      return self.consume_internal(&tokens[n_ctx as usize..tokens.len()]);
    }

    let n_past = if self.n_past + new_tokens > n_ctx {
      n_ctx - new_tokens
    } else {
      self.n_past
    };
    self.context.consume_tokens(tokens, n_past);
    self.n_past = n_past + new_tokens;
    self.n_last = new_tokens;
  }

  pub fn consume_tokens<T>(mut self, tokens: T) -> Self where T: AsRef<[llama_token]> {
    self.consume_internal(tokens.as_ref());
    self
  }

  pub fn consume<P>(mut self, prompt: P) -> Self where P: AsRef<str> {
    let tokens = self.context.tokenize_internal(prompt.as_ref());
    self.consume_internal(&tokens);
    self
  }

  pub fn consume_bos(self) -> Self {
    self.consume_tokens(&[LLaMaCPP::bos_token()])
  }

  pub fn with_sampler_params(mut self, sampler_params: SamplerParams) -> Self {
    self.sampler = Sampler::BuiltinSampler(sampler_params);
    self
  }

  pub fn with_sampler_fn(mut self, sampler_fn: fn(&[f32]) -> llama_token) -> Self {
    self.sampler = Sampler::SamplerFn(sampler_fn);
    self
  }

  pub fn sample(&self) -> llama_token {
    match self.sampler {
      Sampler::BuiltinSampler(params) => self.context.sample_token(params),
      Sampler::SamplerFn(f) => {
        if self.n_last == 0 {
          // TODO: When is this the case during normal usage? Is this arbitrarily decided
          // behavior fine?
          self.context.eos()
        } else {
          let logits = self.context.get_last_logits(self.n_last as usize);
          f(logits)
        }
      },
    }
  }

  pub fn logits(&self) -> Option<&[f32]> {
    if self.n_last == 0 {
      None
    } else {
      Some(self.context.get_last_logits(self.n_last as usize))
    }
  }

  pub fn embeddings(&self) -> Option<&[f32]> {
    self.context.get_embeddings()
  }
}

impl Iterator for LLaMaCPPTokenIter<'_> {
  type Item = String;

  fn next(&mut self) -> Option<Self::Item> {
    let token = self.sample();
    if token == LLaMaCPP::eos_token() {
      return None;
    }
    self.consume_internal(&[token]);
    let token = self.context.token_to_str(token);
    // TODO: Avoid this allocation if possible
    Some(token.to_string())
  }
}

impl LLaMa for LLaMaCPP {
  type Token = llama_token;
  type TokenIterator<'a> = LLaMaCPPTokenIter<'a>;

  fn tokenize(&self, text: &str) -> Vec<Self::Token> {
    self.tokenize_internal(text)
  }

  fn to_str(&self, token: Self::Token) -> &str {
    self.token_to_str(token)
  }

  fn bos(&self) -> Self::Token {
    LLaMaCPP::bos_token()
  }

  fn eos(&self) -> Self::Token {
    LLaMaCPP::eos_token()
  }

  fn iter(&mut self) -> Self::TokenIterator<'_> {
    let sampler = Sampler::BuiltinSampler(SamplerParams::default());
    LLaMaCPPTokenIter { context: self, sampler, n_past: 0, n_last: 0 }
  }
}