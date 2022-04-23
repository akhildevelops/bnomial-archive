use std::{fs::OpenOptions, io::Write};

pub trait ToFileContent {
    fn to_string(&self) -> String;
}

pub fn write_to_file<T: ToFileContent>(x: T, file_path: &str) -> std::io::Result<()> {
    let text = x.to_string();
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(file_path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}
