use bnomial::{today_question, write_to_file};
fn main() {
    let resp = today_question().unwrap();
    write_to_file(resp, "questions.md").unwrap();
}
