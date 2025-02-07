
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
}

document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll("cite[src]").forEach(cite => {
        let sup = document.createElement("sup");
        let link = document.createElement("a");

        link.href = cite.getAttribute("src");
        link.target = "_blank";
        link.innerHTML = "src";

        sup.appendChild(link);
        cite.replaceWith(sup);
    });
});

document.addEventListener("DOMContentLoaded", function () {
  const anchorDiv = document.getElementById("anchors");
  const sections = document.querySelectorAll("section[anchor]");

  let listHTML = "<ul>";

  sections.forEach((section) => {
    const anchorName = section.getAttribute("anchor");
    section.id = anchorName; // Set id to allow linking

    // Extract the original text content and clear the section
    const sectionName = section.textContent.trim();
    section.innerHTML = `<h3>${sectionName}</h3>`;

    // Create a list item with a link
    listHTML += `<li><a href="#${anchorName}">${sectionName}</a></li>`;
  });

  listHTML += "</ul>";
  anchorDiv.innerHTML = listHTML;
});
