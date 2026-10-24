namespace Marmotte.Cli

open System
open System.Globalization
open Spectre.Console
open Spectre.Console.Rendering

/// Composable console helpers. Table modifiers update and return the same table;
/// create a fresh table for each report. Titles, headers, and cells are literal text.
module internal ConsoleDsl =
    /// Create a table with rounded borders and literal column headers.
    let table (columns: string list) =
        let result = Table().RoundedBorder()
        for column in columns do
            result.AddColumn(TableColumn(Text(column))) |> ignore
        result

    /// Append a row without interpreting its cells as Spectre markup.
    let row (values: string list) (table: Table) =
        let cells = values |> List.map (fun value -> Text(value) :> IRenderable) |> List.toArray
        table.AddRow(cells) |> ignore
        table

    /// Append rows in enumeration order.
    let rows (values: seq<string list>) (table: Table) =
        for cells in values do
            row cells table |> ignore
        table

    /// Hide column headers, for example in a property/value summary.
    let hideHeaders (table: Table) =
        table.HideHeaders()

    /// Write a titled section followed by a blank line.
    let writeSection (console: IAnsiConsole) title (content: IRenderable) =
        console.Write(Rule(Markup.Escape title).LeftJustified())
        console.Write(content)
        console.WriteLine()

    /// Limit source characters before escaping control characters for console output.
    let previewText limit (value: string) =
        let prefix = value.Substring(0, min limit value.Length)
        let escaped =
            prefix
            |> String.collect (function
                | '\n' -> "\\n"
                | '\r' -> "\\r"
                | '\t' -> "\\t"
                | character when Char.IsControl character -> sprintf "\\u%04x" (int character)
                | character -> string character)
        if value.Length > limit then escaped + "…" else escaped

    /// Format a byte count using binary units and invariant numeric formatting.
    let formatBytes (bytes: decimal) =
        let units = [| "B"; "KiB"; "MiB"; "GiB"; "TiB"; "PiB"; "EiB" |]
        let mutable size = bytes
        let mutable unitIndex = 0
        while size >= 1024M && unitIndex < units.Length - 1 do
            size <- size / 1024M
            unitIndex <- unitIndex + 1
        sprintf "%s %s" (size.ToString("N2", CultureInfo.InvariantCulture)) units[unitIndex]
