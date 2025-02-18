
function updatepage(linkdetails) {
    const webHeader = document.getElementById("webheader");
    webHeader.innerHTML = `
        <h1>Learning Lattice</h1>
        <i>A compendium of lattice models, gauge theory, and computational physics.</i>
        <br>
        <i>&mdash;&mdash; by Atis Yosprakob</i>
        <br>
        <br>
    `;
    const link1 = document.getElementById("link1");
    const link2 = document.getElementById("link2");
    if(link1){
        link1.innerHTML = linkdetails   
    }
    if(link2){
        link2.innerHTML = linkdetails
    }
    box_embed()
}

function box_embed(){
    document.body.innerHTML = document.body.innerHTML.replace(/\$\$(.*?)\$\$/gs, function(match, content) {
                return `<div class="equation">\\begin{align}${content}\\end{align}</div>`;
            });
}

document.addEventListener("DOMContentLoaded", function() {

    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Manually revise some contents ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    // Custom tags
    var contentText = document.getElementById("content").innerHTML

    // new paragraph
    contentText = contentText.replaceAll("<nl>", "<br><br>");

    // textbox
    contentText = contentText.replaceAll("<box>", "<div class='text-box'>");
    contentText = contentText.replaceAll("</box>", "</div>");

    contentText = contentText.replaceAll("</note>", "<sup>[?]</sup></note>");

    //comment
    contentText = contentText.replaceAll("/*", "<!--");
    contentText = contentText.replaceAll("*/", "-->");

    //ref to other page
    contentText = contentText.replaceAll("<ref ", "<a target='_blank' ");
    contentText = contentText.replaceAll("</ref", "</a");

    document.getElementById("content").innerHTML = contentText;

    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Citation :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    document.querySelectorAll("cite[src]").forEach(cite => {
        let sup = document.createElement("sup");
        let link = document.createElement("a");

        link.href = cite.getAttribute("src");
        link.target = "_blank";
        link.innerHTML = "src";

        sup.appendChild(link);
        cite.replaceWith(sup);
    });

    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Section symbols ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    const anchorDiv = document.getElementById("anchors");
    const sections = document.querySelectorAll("section[anchor]");

    let listHTML = "<ol>";

    sections.forEach((section,index) => {
    const anchorName = section.getAttribute("anchor");
    section.id = anchorName; // Set id to allow linking

    // Extract the original text content and clear the section
    const sectionName = section.textContent.trim();
    section.innerHTML = `<h3>${index+1}. ${sectionName}</h3>`;

    // Create a list item with a link
    listHTML += `<li><a href="#${anchorName}">${sectionName}</a></li>`;
    });

    listHTML += "</ol>";
    anchorDiv.innerHTML = listHTML;
    

    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Repositioning tooltips :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    function positionTooltip(note) {
        const rect = note.getBoundingClientRect();
        const screenWidth = window.innerWidth;
        const noteCenter = rect.left + rect.width / 2;

        if (noteCenter < screenWidth / 2) {
            note.style.setProperty('--tooltip-a', '0%');
            note.style.setProperty('--tooltip-b', '0%');
        } else {
            note.style.setProperty('--tooltip-a', '-100%');
            note.style.setProperty('--tooltip-b', '100%');
        }
    }
    
    document.querySelectorAll('note').forEach(note => {
        note.addEventListener('mouseenter', function() {positionTooltip(note);});
    });

    // Close tooltips when clicking outside
    document.addEventListener('click', function() {
        document.querySelectorAll('.tooltip-visible').forEach(el => el.classList.remove('tooltip-visible'));
    });
    
});

