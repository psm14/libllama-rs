use libllama::{ LLM, LLaMA };
use std::env::args;
use std::io;
use std::io::Write;

fn main() {
  let file = args().nth(1).expect("Missing model file (first argument)");

  let mut params = LLaMA::default_params();
  params.n_ctx = 2048;
  let mut llama = LLaMA::from_file(&file, params).expect("Failed to load model");

  let prompt = args().nth(2).expect("Missing prompt (second argument)");
  print!("{}", prompt);
  io::stdout().flush().unwrap();

  for token in llama.iter().consume_bos().consume(" ").consume(prompt) {
    print!("{}", token);
    io::stdout().flush().unwrap();
  }

  println!("");
}
