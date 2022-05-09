use crate::defaults;
use crate::write_file::ToFileContent;
use chrono::prelude::*;
use reqwest;
use serde::{de, Deserialize, Deserializer};
use std::collections::HashMap;

macro_rules! markdown {
    () => {
        "<details><summary>CLICK ME</summary><p>{}</p></details>"
    };
}

fn naive_date_time_from_str<'de, D>(deserializer: D) -> Result<NaiveDate, D::Error>
where
    D: Deserializer<'de>,
{
    let s = Deserialize::deserialize(deserializer)?;
    NaiveDate::parse_from_str(s, "%Y%m%d").map_err(de::Error::custom)
}

#[derive(Deserialize)]
pub struct BnomialRespContent {
    #[serde(deserialize_with = "naive_date_time_from_str")]
    date: NaiveDate,
    title: String,
    content: String,
    choices: Vec<String>,
    #[serde(default)]
    answer: String,
    #[serde(default)]
    explanation: String,
    #[serde(default)]
    references: String,
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
        let answer = format!(markdown!(), self.answer);
        let explaination = format!(markdown!(), self.explanation);
        let references = format!(markdown!(), self.references);
        format!(
            "## Date - {}\n\n\n## Title - {}\n\n\n### **Question** :\n\n{}\n\n\n### **Choices** :\n\n{}\n\n\n### **Answer** :\n\n{}\n\n\n### **Explaination** :\n\n{}\n\n\n### **References**: \n\n{}\n\n-----------------------\n\n",
            self.date, self.title, self.content, choices, answer, explaination, references
        )
    }
}

fn bnomial_rand_answer(n_choices: usize) -> String {
    let mut answer = "0".repeat(n_choices - 1);
    answer.push('1');
    answer
}

pub fn today_question() -> Result<BnomialRespContent, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let response_content = client
        .get(defaults::QUESTION_URL)
        .send()?
        .json::<BnomialResp>()?
        .question;

    let random_answer = bnomial_rand_answer(response_content.choices.len());

    let mut map = HashMap::new();
    map.insert("answer", random_answer);

    let response = client
        .post(defaults::ANSWER_URL)
        .json(&map)
        .send()?
        .json::<BnomialResp>()?;
    Ok(response.question)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post() {
        let mut map = HashMap::new();
        map.insert("answer", "0000");
        let client = reqwest::blocking::Client::new();
        let _response = client.post(defaults::ANSWER_URL).json(&map).send().unwrap();
    }
}
