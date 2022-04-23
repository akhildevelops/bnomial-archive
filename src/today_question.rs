use crate::defaults;
use crate::write_file::ToFileContent;
use reqwest;
use serde::Deserialize;
#[derive(Deserialize)]
pub struct BnomialRespContent {
    date: String,
    title: String,
    content: String,
    choices: Vec<String>,
}

#[derive(Deserialize)]
struct BnomialResp {
    question: BnomialRespContent,
}

impl ToFileContent for BnomialRespContent {
    fn to_string(&self) -> String {
        let choices = self
            .choices
            .iter()
            .map(|x| "- ".to_owned() + x)
            .reduce(|x, y| [x, y].join("\n"))
            .unwrap();
        format!(
            "## Date - {}\n\n\n## Title - {}\n\n\n### **Question** :\n\n{}\n\n\n### **Choices** :\n\n{}\n\n-----------------------\n\n",
            self.date, self.title, self.content, choices
        )
    }
}

pub fn today_question() -> Result<BnomialRespContent, Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(defaults::URL)?.json::<BnomialResp>()?;
    Ok(response.question)
}
