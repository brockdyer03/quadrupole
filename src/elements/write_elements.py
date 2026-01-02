from pathlib import Path
from urllib import request
from html.parser import HTMLParser
from datetime import datetime


#iupac_url = "https://ciaaw.org/abridged-atomic-weights.htm"
alt_iupac_url = "https://iupac.qmul.ac.uk/AtWt/"
#request.urlretrieve(iupac_url, "iupac_atomic_weights.html")
request.urlretrieve(alt_iupac_url, "alt_iupac_element_data.html")

class Alt_IUPAC_HTMLParser(HTMLParser):
    """Rudimentary HTML Parser for scraping the atomic weights out of the alternate IUPAC Website"""
    def __init__(self, table_identifier: str):
        HTMLParser.__init__(self)
        self.table_identifier = table_identifier
        self.found_table = False
        self.end_table = False
        self.data = {}
        self.start_initialize = False
        self.table_initialized = False
        self.data_keys = []
        self.data_col = 0
        self.in_data_entry = False

    def handle_starttag(self, tag, attrs):

        if self.found_table and not self.table_initialized:
            # Begin initializing the table header
            if tag == "td" and not self.start_initialize:
                self.start_initialize = True
                
        elif self.table_initialized:
            if tag == "td":
                # Indicate we are in a data entry
                self.in_data_entry = True
            

    def handle_endtag(self, tag):
        if self.found_table and not self.end_table:
            if tag == "table":
                # Indicate table has ended
                self.end_table = True
            elif self.start_initialize and not self.table_initialized:
                if tag == "tr":
                    self.table_initialized = True
                    self.data_col = 0
                    #print(f"Data table initialized with these keys: {self.data_keys}")
            if tag == "td":
                # Advance column if we have hit the end of the <td> block
                self.in_data_entry = False
                self.data_col = self.data_col + 1
            elif tag == "tr":
                self.data_col = 0
            

    def handle_data(self, data):
        data = data.strip()
        if data == self.table_identifier:
            self.found_table = True

        if len(data) != 0 and self.found_table and not self.end_table:
            if self.start_initialize and not self.table_initialized:
                # Initialize the data table with the headers
                self.data[data] = []
                self.data_keys.append(data)
            elif self.table_initialized:
                key = self.data_keys[self.data_col]
                self.data[key] += [data]

with open("./alt_iupac_element_data.html", "r") as iupac:
    alt_iupac_html = iupac.readlines()

parser = Alt_IUPAC_HTMLParser(table_identifier="Table 2. List of Elements in Atomic Number Order.")
for line in alt_iupac_html:
    parser.feed(line)
    if parser.end_table:
        break

alt_iupac_data = {}
for symbol in parser.data["Symbol"]:
    alt_iupac_data[symbol] = {}

for key in parser.data:
    if key in ["Symbol", "Notes"]:
        continue

    data = zip(parser.data["Symbol"], parser.data[key])
    for symbol, datum in data:
        if key == "At No":
            alt_iupac_data[symbol]["atomic_number"] = int(datum)
        elif key == "Name":
            alt_iupac_data[symbol]["name"] = datum.title()
        elif key == "Atomic Wt":
            datum = datum.replace(" ", "")
            datum = datum.replace("[", "").replace("]", "").replace("(", " ")
            datum = datum.split()[0]
            if datum.isdigit():
                datum += ".0"
            alt_iupac_data[symbol]["atomic_weight"] = datum


toml_output = Path("./iupac_atomic_weights.toml")

with open(toml_output, "w") as out:
    out.write("[[element]]\n")
    for element in alt_iupac_data:
        dat = alt_iupac_data[element]
        line = ""
        line += f"[element.{element}]\n"
        for key in dat:
            datum = dat[key]
            if key == "name":
                line += f'{key:13} = "{datum}"\n'
            else:
                line += f"{key:13} = {datum}\n"
        line += "\n"

        out.write(line)

elements = {
    "Xx": {
        "atomic_number": 0,
        "name": "Unknown",
        "atomic_weight": 0.0
    }
}
elements.update(alt_iupac_data)
elements["Al"]["name"] = "Aluminum" # Switch to American english spelling
elements["Cs"]["name"] = "Cesium"   # (Add freedom)

rust_output_file = Path(__file__+"/../../elements.rs").resolve()
rust_element_header = (
    f"// Sources used for code:\n"
    f"// {alt_iupac_url}\n\n"
    f"// Data Retrieved on {datetime.now()}\n"
    f"// Code was written automatically by src/elements/write_elements.py\n\n"
    "use strum_macros;\n\n"
)

rust_code = ""
rust_code += rust_element_header

rust_code += "#[derive(Debug, PartialEq, Eq, Clone, Copy, strum_macros::EnumString, strum_macros::Display, strum_macros::AsRefStr)]\n"
rust_code += "pub enum Element {\n"
for element in elements:
    rust_code += f"    {element},\n"
rust_code += "}\n"

rust_code += (
"""
impl Element {

    pub fn symbol(&self) -> &str {
        self.as_ref()
    }

    pub fn name(&self) -> &str {
        match *self {
"""
)
for element in elements:
    name = elements[element]["name"]
    rust_code += " "*12
    rust_code += f'Element::{element:2} => "{name}",\n'
rust_code += "      }\n"
rust_code += "    }\n"

rust_code += (
"""
    pub fn number(&self) -> u8 {
        *self as u8
    }

    pub fn weight(&self) -> f32 {
        match *self {
"""
)

for element in elements:
    weight = elements[element]["atomic_weight"]
    rust_code += " "*12
    rust_code += f"Element::{element:2} => {weight},\n"
rust_code += "        }\n"
rust_code += "    }\n"
rust_code += "}\n"

rust_code += (
"""
static ELEMENTS: [Element; 119] = [
"""
)
for element in elements:
    rust_code += " "*4
    rust_code += f"Element::{element},\n"
rust_code += "];\n"

rust_code += (
"""
impl TryFrom<usize> for Element {
    type Error = ();

    fn try_from(number: usize) -> Result<Self, Self::Error> {
        if number < ELEMENTS.len() {
            return Ok(ELEMENTS[number])
        } else {
            Err(())
        }
    }
}
"""
)

with open(rust_output_file, "w") as rs_out:
    rs_out.write(rust_code)