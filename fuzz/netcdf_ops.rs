//! Turn fuzzer bytes into calls of the netCDF writer and compare the file with a model.
//!
//! Each call can return an error. An error must change nothing. A call that
//! returns `Ok` changes the model. At the end the reader must give the model.
//!
//! The model has the root group and its child groups. It does not add a
//! dimension that hides a dimension of the root group from a variable that
//! uses it, because the reader finds a dimension by its name.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use oxcdf::hdf5::message::ByteOrder;
use oxcdf::{DType, FileMut, MemorySource, NcAttribute, NcDimension, NcVariable, NetcdfFile};

/// Stop one input from a long run.
const MAX_OPS: usize = 48;

/// Largest number of values in one write.
const MAX_VALUES: u64 = 64;

/// Names of dimensions, variables and groups. Some repeat, and the last four are wrong.
const NAMES: [&str; 12] = [
    "a", "b", "c", "t", "x", "y", "lat", "é", "", "p/q", " z", "-n",
];

/// Attribute names. The last four are wrong or reserved.
const ATTRIBUTES: [&str; 8] = [
    "units",
    "title",
    "n",
    "range",
    "",
    "CLASS",
    "_FillValue",
    "NAME",
];

/// Text values. One is empty.
const TEXTS: [&str; 4] = ["", "m", "degC", "Den Helder"];

const DTYPES: [DType; 8] = [
    DType::Int(1),
    DType::Int(2),
    DType::Int(4),
    DType::Int(8),
    DType::Uint(2),
    DType::Float(4),
    DType::Float(8),
    DType::String,
];

/// The fuzzer bytes, read from the front.
struct Input<'a>(&'a [u8]);

impl Input<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&first, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(first)
    }

    /// The next byte as an index below `len`. A short input gives 0.
    fn index(&mut self, len: usize) -> usize {
        usize::from(self.byte().unwrap_or(0)) % len.max(1)
    }
}

/// The value of one element.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Number(f64),
    Text(String),
}

#[derive(Debug)]
struct Dimension {
    name: String,
    len: u64,
    unlimited: bool,
}

#[derive(Debug)]
struct Variable {
    name: String,
    dtype: DType,
    /// The group and the position of the dimension of each axis.
    dimensions: Vec<(usize, usize)>,
    /// Each element that has a write, by its index.
    cells: BTreeMap<Vec<u64>, Cell>,
    fill: Cell,
    /// Name and debug text of each attribute, in put order.
    attributes: Vec<(String, String)>,
}

#[derive(Debug, Default)]
struct Group {
    /// `None` for the root group.
    name: Option<String>,
    dimensions: Vec<Dimension>,
    variables: Vec<Variable>,
    attributes: Vec<(String, String)>,
}

/// Call a method of the group `$group` of `$file`. The root group and a child have the same methods.
macro_rules! on {
    ($file:expr, $group:expr, $($call:tt)+) => {
        match $group {
            None => $file.$($call)+,
            Some(name) => $file
                .group_mut(name)
                .expect("the model has the group")
                .$($call)+,
        }
    };
}

/// The default fill value of netCDF for `dtype`, as the reader gives it for `f64`.
fn default_fill(dtype: &DType) -> Cell {
    Cell::Number(match dtype {
        DType::Int(1) => -127.0,
        DType::Int(2) => -32767.0,
        DType::Int(4) => -2_147_483_647.0,
        DType::Int(8) => -9_223_372_036_854_775_806_i64 as f64,
        DType::Uint(2) => 65535.0,
        DType::Float(4) => f64::from(9.969_21e36_f32),
        DType::Float(8) => 9.969_209_968_386_869e36,
        _ => return Cell::Text(String::new()),
    })
}

/// Put `value` under `name`, at its first position when the name exists.
fn put(attributes: &mut Vec<(String, String)>, name: &str, value: String) {
    match attributes.iter_mut().find(|(n, _)| n == name) {
        Some(entry) => entry.1 = value,
        None => attributes.push((name.to_owned(), value)),
    }
}

/// The groups of the model. The root group is at 0, and each other group is its child.
struct Model(Vec<Group>);

impl Model {
    /// The group and position of the dimension `name` that a variable of `group` sees.
    fn find_dimension(&self, group: usize, name: &str) -> Option<(usize, usize)> {
        [group, 0].into_iter().find_map(|g| {
            let position = self.0[g].dimensions.iter().position(|d| d.name == name)?;
            Some((g, position))
        })
    }

    /// The shape that the reader reports for `variable`.
    fn shape(&self, variable: &Variable) -> Vec<u64> {
        let dimension = |&(g, d): &(usize, usize)| self.0[g].dimensions[d].len;
        variable.dimensions.iter().map(dimension).collect()
    }

    /// Whether a new dimension `name` in `group` hides a root dimension that a variable uses.
    fn hides(&self, group: usize, name: &str) -> bool {
        let Some(root) = self.0[0].dimensions.iter().position(|d| d.name == name) else {
            return false;
        };
        let uses = |v: &Variable| v.dimensions.contains(&(0, root));
        group != 0 && self.0[group].variables.iter().any(uses)
    }
}

/// One attribute put on the file, a group or a variable. Returns the debug text of the value.
fn put_attribute(
    input: &mut Input<'_>,
    file: &mut FileMut,
    group: Option<&str>,
    variable: Option<&str>,
    name: &str,
) -> Option<String> {
    macro_rules! send {
        ($value:expr) => {
            match variable {
                None => on!(file, group, add_attribute(name, $value)),
                Some(v) => on!(
                    file,
                    group,
                    variable_mut(v)
                        .expect("the model has the variable")
                        .put_attribute(name, $value)
                ),
            }
        };
    }
    match input.index(3) {
        0 => {
            let text = TEXTS[input.index(TEXTS.len())];
            send!(text).ok().map(|()| format!("Str({text:?})"))
        }
        1 => {
            let count = input.index(4);
            let values: Vec<i32> = (0..count).map(|_| input.index(200) as i32 - 100).collect();
            let text = match values.as_slice() {
                [one] => format!("Int({one})"),
                many => format!("Ints({many:?})"),
            };
            send!(values).ok().map(|()| text)
        }
        _ => {
            let value = input.index(200) as f64 * 0.25;
            send!(value).ok().map(|()| format!("Double({value:?})"))
        }
    }
}

/// Change the definition of a variable. Returns the new fill value after a fill value call.
fn redefine(
    input: &mut Input<'_>,
    file: &mut FileMut,
    group: Option<&str>,
    name: &str,
    rank: usize,
) -> Option<Cell> {
    macro_rules! variable {
        ($($call:tt)+) => {
            on!(
                file,
                group,
                variable_mut(name).expect("the model has the variable").$($call)+
            )
        };
    }
    match input.index(6) {
        0 => {
            // Each type of the model holds a value from 0 to 99.
            let fill = input.index(100) as i32;
            let set = variable!(set_fill_value(fill)).is_ok();
            return set.then(|| Cell::Number(f64::from(fill)));
        }
        1 => {
            let chunks: Vec<usize> = (0..rank).map(|_| input.index(4)).collect();
            let _ = variable!(set_chunking(&chunks));
        }
        2 => {
            let level = input.index(11) as i32;
            let shuffle = input.index(2) == 1;
            let _ = variable!(set_compression(level, shuffle));
        }
        3 => {
            let _ = variable!(set_zstd(input.index(24) as i32));
        }
        4 => {
            let _ = variable!(set_fletcher32(input.index(2) == 1));
        }
        _ => {
            let order = [ByteOrder::Little, ByteOrder::Big][input.index(2)];
            let _ = variable!(set_endianness(order));
        }
    }
    None
}

/// Each index of the selection `ranges`, in row-major order.
fn indices(ranges: &[Range<usize>]) -> Vec<Vec<u64>> {
    let mut all = vec![Vec::new()];
    for range in ranges {
        all = all
            .into_iter()
            .flat_map(|head| {
                range.clone().map(move |i| {
                    let mut index = head.clone();
                    index.push(i as u64);
                    index
                })
            })
            .collect();
    }
    all
}

/// Run the writer calls that `data` encodes, then compare the file with the model.
pub fn run(data: &[u8]) {
    let mut input = Input(data);
    let mut file = oxcdf::create_in_memory().expect("a file in memory opens");
    let mut model = Model(vec![Group::default()]);

    for _ in 0..MAX_OPS {
        let Some(op) = input.byte() else { break };
        let g = input.index(model.0.len());
        let group_name = model.0[g].name.clone();
        let group = group_name.as_deref();
        match op % 8 {
            0 => {
                let name = NAMES[input.index(NAMES.len())];
                let len = input.index(5);
                if model.hides(g, name) {
                    continue;
                }
                let added = if input.index(4) == 0 {
                    on!(file, group, add_unlimited_dimension(name)).map(|()| 0)
                } else {
                    on!(file, group, add_dimension(name, len)).map(|()| len)
                };
                if let Ok(len) = added {
                    model.0[g].dimensions.push(Dimension {
                        name: name.to_owned(),
                        len: len as u64,
                        unlimited: len == 0,
                    });
                }
            }
            1 => {
                let name = NAMES[input.index(NAMES.len())];
                let dtype = DTYPES[input.index(DTYPES.len())].clone();
                let rank = input.index(4);
                let names: Vec<&str> = (0..rank).map(|_| NAMES[input.index(7)]).collect();
                let added = on!(
                    file,
                    group,
                    add_variable_with_type(name, &names, &dtype).map(|_| ())
                );
                if added.is_ok() {
                    let find = |n: &&str| model.find_dimension(g, n);
                    let dimensions: Option<Vec<_>> = names.iter().map(find).collect();
                    let dimensions = dimensions.expect("the writer found each dimension");
                    model.0[g].variables.push(Variable {
                        name: name.to_owned(),
                        fill: default_fill(&dtype),
                        dtype,
                        dimensions,
                        cells: BTreeMap::new(),
                        attributes: Vec::new(),
                    });
                }
            }
            2 => {
                let name = NAMES[input.index(NAMES.len())];
                if file.add_group(name).is_ok() {
                    model.0.push(Group {
                        name: Some(name.to_owned()),
                        ..Group::default()
                    });
                }
            }
            3 | 4 => {
                if model.0[g].variables.is_empty() {
                    continue;
                }
                let v = input.index(model.0[g].variables.len());
                let variable = &model.0[g].variables[v];
                let name = variable.name.clone();
                let is_text = variable.dtype == DType::String;
                let shape = model.shape(variable);
                // A full selection has the reader shape. Each other one can pass it.
                let full = input.index(4) == 0;
                let ranges: Vec<Range<usize>> = shape
                    .iter()
                    .map(|&len| {
                        let start = input.index(4);
                        let count = input.index(4);
                        match full {
                            true => 0..len as usize,
                            false => start..start + count,
                        }
                    })
                    .collect();
                let count: u64 = ranges.iter().map(|r| r.len() as u64).product();
                if count > MAX_VALUES {
                    continue;
                }
                // A wrong value count must return an error and change nothing.
                let given = match input.index(8) {
                    0 => count + 1,
                    _ => count,
                };
                let numbers: Vec<i32> = (0..given).map(|_| input.index(100) as i32).collect();
                let wrote = match (is_text, op % 8) {
                    (true, _) => {
                        let texts: Vec<String> = numbers.iter().map(|n| format!("s{n}")).collect();
                        on!(
                            file,
                            group,
                            variable_mut(&name)
                                .expect("the model has the variable")
                                .put_strings(&texts, ranges.clone())
                        )
                    }
                    (false, 3) => on!(
                        file,
                        group,
                        variable_mut(&name)
                            .expect("the model has the variable")
                            .put_values(&numbers, ranges.clone())
                    ),
                    (false, _) => {
                        let doubles: Vec<f64> = numbers.iter().map(|&n| f64::from(n)).collect();
                        on!(
                            file,
                            group,
                            variable_mut(&name)
                                .expect("the model has the variable")
                                .put_values(&doubles, ranges.clone())
                        )
                    }
                };
                if wrote.is_err() || count == 0 {
                    continue;
                }
                assert_eq!(given, count, "a write with a wrong value count passes");
                let axes = model.0[g].variables[v].dimensions.clone();
                for (&(dg, d), range) in axes.iter().zip(&ranges) {
                    let dimension = &mut model.0[dg].dimensions[d];
                    assert!(
                        dimension.unlimited || range.end as u64 <= dimension.len,
                        "a write passes the end of a fixed dimension"
                    );
                    dimension.len = dimension.len.max(range.end as u64);
                }
                let cells = &mut model.0[g].variables[v].cells;
                for (index, number) in indices(&ranges).into_iter().zip(numbers) {
                    let cell = match is_text {
                        true => Cell::Text(format!("s{number}")),
                        false => Cell::Number(f64::from(number)),
                    };
                    cells.insert(index, cell);
                }
            }
            5 => {
                if model.0[g].variables.is_empty() {
                    continue;
                }
                let v = input.index(model.0[g].variables.len());
                let variable = &model.0[g].variables[v];
                let (name, rank) = (variable.name.clone(), variable.dimensions.len());
                if let Some(fill) = redefine(&mut input, &mut file, group, &name, rank) {
                    assert!(
                        model.0[g].variables[v].cells.is_empty(),
                        "a fill value changes after a write"
                    );
                    model.0[g].variables[v].fill = fill;
                }
            }
            6 => {
                let name = ATTRIBUTES[input.index(ATTRIBUTES.len())];
                if let Some(value) = put_attribute(&mut input, &mut file, group, None, name) {
                    put(&mut model.0[g].attributes, name, value);
                }
            }
            _ => {
                if model.0[g].variables.is_empty() {
                    continue;
                }
                let v = input.index(model.0[g].variables.len());
                let variable = model.0[g].variables[v].name.clone();
                let name = ATTRIBUTES[input.index(ATTRIBUTES.len())];
                let value = put_attribute(&mut input, &mut file, group, Some(&variable), name);
                if let Some(value) = value {
                    put(&mut model.0[g].variables[v].attributes, name, value);
                }
            }
        }
    }

    let bytes = file.into_bytes().expect("the close passes");
    let file = NetcdfFile::from_source(Arc::new(MemorySource::new(bytes)))
        .expect("the reader opens the file");
    for group in &model.0 {
        check_group(&model, group, &file);
    }
}

/// Name and debug text of each attribute, with no `_FillValue`.
fn attribute_texts(attributes: &[NcAttribute]) -> Vec<(String, String)> {
    attributes
        .iter()
        .filter(|a| a.name != "_FillValue")
        .map(|a| (a.name.clone(), format!("{:?}", a.value)))
        .collect()
}

/// Compare one group of the model with the file.
fn check_group(model: &Model, group: &Group, file: &NetcdfFile) {
    let path = group.name.as_ref().map(|n| format!("/{n}"));
    let (dimensions, variables, attributes): (&[NcDimension], &[NcVariable], &[NcAttribute]) =
        match &path {
            None => (file.dimensions(), &file.root().variables, file.attributes()),
            Some(path) => {
                let group = file.group(path).expect("the file has the group");
                (&group.dimensions, &group.variables, &group.attributes)
            }
        };
    let found: Vec<(&str, u64, bool)> = dimensions
        .iter()
        .map(|d| (d.name.as_str(), d.len, d.is_unlimited))
        .collect();
    let expected: Vec<(&str, u64, bool)> = group
        .dimensions
        .iter()
        .map(|d| (d.name.as_str(), d.len, d.unlimited))
        .collect();
    assert_eq!(found, expected, "dimensions of {path:?}");
    assert_eq!(
        attribute_texts(attributes),
        group.attributes,
        "attributes of {path:?}"
    );

    let names: Vec<&str> = variables.iter().map(|v| v.name.as_str()).collect();
    let expected: Vec<&str> = group.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, expected, "variables of {path:?}");

    for variable in &group.variables {
        let full = format!("{}/{}", path.as_deref().unwrap_or(""), variable.name);
        let found = file.variable(&full).expect("the file has the variable");
        let shape = model.shape(variable);
        assert_eq!(found.shape, shape, "shape of {full}");
        let names: Vec<&str> = variable
            .dimensions
            .iter()
            .map(|&(g, d)| model.0[g].dimensions[d].name.as_str())
            .collect();
        assert_eq!(found.dimensions, names, "dimensions of {full}");
        assert_eq!(found.vartype(), variable.dtype, "type of {full}");
        assert_eq!(
            attribute_texts(found.attributes()),
            variable.attributes,
            "attributes of {full}"
        );

        let ranges: Vec<Range<usize>> = shape.iter().map(|&len| 0..len as usize).collect();
        let expected: Vec<Cell> = indices(&ranges)
            .iter()
            .map(|index| variable.cells.get(index).unwrap_or(&variable.fill).clone())
            .collect();
        let values: Vec<Cell> = if variable.dtype == DType::String {
            let texts = found.get_strings(..).expect("the strings read");
            texts.into_iter().map(Cell::Text).collect()
        } else {
            let numbers = found.get_values::<f64, _>(..).expect("the values read");
            numbers.into_iter().map(Cell::Number).collect()
        };
        assert_eq!(values, expected, "values of {full}");
    }
}
