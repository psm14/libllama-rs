mod llama;

use llama::{ LLaMa, LLaMaCPP, SamplerParams };
use std::env::args;
use std::io;
use std::io::Write;

fn main() {
  let file = args().nth(1).expect("Missing model file");

  let mut params = LLaMaCPP::default_params();
  params.n_ctx = 2048;
  let mut llama = LLaMaCPP::from_file(&file, params).expect("Failed to load model");

  let prompt = "### Instruction:\nYou are ChatLLaMa, an honest and helpful chatbot\n\n### Input:\nHow tall is the Eiffel Tower?\n\n### Response:\n";
  print!("{}", prompt);
  io::stdout().flush().unwrap();

  for token in llama.token_iter(SamplerParams::default()).consume_bos().consume(&prompt) {
    print!("{}", token);
    io::stdout().flush().unwrap();
  }

  println!("");
}
