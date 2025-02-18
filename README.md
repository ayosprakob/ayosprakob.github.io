
# Custom syntax for the html page

### Section
Use `<section anchor="section_name">Section name</section>` to begin a new section.

### New paragraph
Use `<nl>` to begin a new paragraph. It's the same as using `<br><br>`

### Citation
Use `<cite src="link.html"></cite>` to make a link to the citation.

### Text box
Use `<box>Content...</box>` to make a box containing the text.

### Footnote
Use `<note value="footnote content">text to mark the footnote</note>` to make a footnote over the text.

### Comment
Use `/* Comment. */` comment out a block. Can be multi-line.

### Link to other page
Use `<ref href="target.html">text</ref>` for the link. Similar to `<ref href="target.html" target="_blank" >text</ref>`

### Table
Use the following format
```
    <mytable>
      <trow>
        <hcol>header1
        <hcol>header2
        <hcol>header3
        <hcol>header4
      <trow>
        <tcol>This
        <tcol>is
        <tcol>the
        <tcol>test.
      <trow>
        <tcol>Here
        <tcol>is
        <tcol>another
        <tcol>row.
    </mytable>
```