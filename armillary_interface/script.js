
function ClearStep2(){
    document.getElementById("output2").innerHTML = "";
    ClearStep3();
}

function ClearStep3(){
    document.getElementById('step3title').innerHTML = ""
    document.getElementById("SUN").innerHTML = "";
    document.getElementById("characterExpansion").innerHTML = "";
    document.getElementById("Step4Button").innerHTML = "";
    ClearStep4();
}

function ClearStep4(){
    document.getElementById("output4").innerHTML = "";
    document.getElementById("Step5Button").innerHTML = "";
    ClearStep5();
}

function ClearStep5(){
    document.getElementById("output5").innerHTML = "";
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//         Pyodine preparation
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

let python_result = ""; // Global variable for Python output
let pyodideInstance = null; // Pyodide instance placeholder
let pyodideReady = false; // Flag to track initialization status

// Function to initialize Pyodide
async function initializePyodide() {
    if (!pyodideInstance) {
        console.log("Initializing Pyodide...");
        pyodideInstance = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.3/full/"
        });

        console.log("Loading numpy...");
        await pyodideInstance.loadPackage("numpy"); // Ensure numpy is fully loaded
        console.log("Numpy loaded!");

        // Redirect print() output to our global variable
        pyodideInstance.globals.set("print", (text) => {
            python_result += ""+text + "<br>";
        });

        pyodideReady = true;
        console.log("Pyodide is fully ready.");
    }

    // Wait if Pyodide is not ready yet (prevents race conditions)
    while (!pyodideReady) {
        await new Promise(resolve => setTimeout(resolve, 10)); // Wait 10ms before checking again
    }
}

// Run Python function
async function run_python(args, raw_code) {
    await initializePyodide(); // Make sure Pyodide is fully initialized
    python_result = ""
    let pythonCode = raw_code;
    for (let key in args) {
        pythonCode = pythonCode.replaceAll(`__${key}__`, args[key]);
    }

    try {
        await pyodideInstance.runPythonAsync(pythonCode);
    } catch (err) {
        python_result = 'Error: ' + err;
    }
}

// Automatically initialize Pyodide on script load
initializePyodide();

// Expose `run_python` globally so it can be called from anywhere
window.run_python = run_python;


// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//         Step 1
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function renderMath() {
    let input = document.getElementById("latexInput").value;
    if(input.length>0){
        input = latexReformat(input)
        if(input[0]=="+"){
            input = input.replace("+","")
        }
        document.getElementById("output1").innerHTML = "<div class=\"wrap\">\\[" + input + "\\]</div>";
        MathJax.typesetPromise();
        //countUOccurrences();
        ClearStep2();
    }
}

function applyPresets() {
    let textArea = document.getElementById("latexInput");
    let checkboxes = document.querySelectorAll("input[type=checkbox]:checked");
    let presetText = "";
    
    checkboxes.forEach(box => {
        presetText += box.value + "\n\n";
    });
    
    textArea.value = presetText;
    adjustHeight(textArea);
}

function adjustHeight(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
}

function ordinal(x) {
    if (typeof x !== "number" || !Number.isInteger(x)) {
        throw new Error("Input must be an integer.");
    }

    let suffix = "th";
    if (x % 100 < 11 || x % 100 > 13) {
        switch (x % 10) {
            case 1: suffix = "st"; break;
            case 2: suffix = "nd"; break;
            case 3: suffix = "rd"; break;
        }
    }

    return x + suffix;
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//         Step 2
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
let gbLatexExpr
let gbParsed_operator
let gbParsed_sum
let gbPairs
let gbLinkCount
function countUOccurrences() {
    ClearStep3();

    let display = "Please confirm that everything is correct. If not, check your input again."

    let latexExpr = document.getElementById("latexInput").value;
    latexExpr = latexReformat(latexExpr)
    latexExpr = latexExpr.replaceAll("-","+-")
    latexExpr = latexExpr.split("+").slice(1)

    

    let term = 1
    let linkCount = []
    let parsed_operator = []
    let parsed_sum = []
    let pairs = []
    let linkCountByTerm = []
    for(const expression of latexExpr){
        let result = parseOneTerm(expression)
        let linkCountThisTerm = []
        for(let key in result[1]){
            if(key in linkCount)
                linkCount[key] += result[1][key]+result[2][key]
            else
                linkCount[key] = result[1][key]+result[2][key]

            if(key in linkCountThisTerm)
                linkCountThisTerm[key] += result[1][key]+result[2][key]
            else
                linkCountThisTerm[key] = result[1][key]+result[2][key]
        }
        linkCountByTerm.push(linkCountThisTerm)
        if(result[0]==="")
            continue
        display += ("<h4>$\\bullet$ "+ordinal(term)+" term "
            +"<font color='#888'>"
            +"::::::::::::::::::::::::::::::::::::"
            +"</font>"
            +"</h4>")
        //if(parsed_operator.includes(result[2]))
        //    display+= "<font color='red'>Note: this term can and should be merged with one of the previous terms.</font><br>"
        display += result[0]
        parsed_operator.push(result[3])
        parsed_sum.push(result[4])
        pairs.push(result[5])
        term+=1
    }
    if(display!=""){
        display += ("<h4>$\\bullet$ Summary "
            +"<font color='#888'>"
            +"&mdash;&mdash;&mdash;&mdash;&mdash;"
            +"&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;"
            +"</font>"
            +"</h4>")
        display += "Total counts per site:  <br> "
        display += "&nbsp;&nbsp;&nbsp;&nbsp;"
        let ilink = 0
        for(let key in linkCount){
            let count = linkCount[key]
            if(count>0){
                if(ilink>0)
                    display += ", "
                display += "$"+latexReformat(key)+"$&times;<font color='blue'>"+count+"</font>"
                ilink+=1
            }
            
        }
        display += "<nl>"
    }
    display += `<br><br>
    <button onclick="createSUNDropdown();">To Step 3</button>
    <br>`

    let notice_text = ""//"<i><font color='blue'>Note: If the result looks strange, the reason might be that the input is ambiguous.</font></i><nl>"
    let title = "<br><hr><h2>Step 2: Count the link variables</h2>"
    document.getElementById("output2").innerHTML = title+notice_text+display;
    MathJax.typesetPromise();

    gbLatexExpr = latexExpr
    gbParsed_operator = parsed_operator
    gbParsed_sum = parsed_sum
    gbPairs = pairs
    gbLinkCount = linkCountByTerm

    //createSUNDropdown()
}

function latexReformat(latexExpr){
    let tr = "\\text{Tr}"
    let re = "\\text{Re}"
    let im = "\\text{Im}"
    latexExpr = latexExpr.replaceAll("\\tr",tr)
    latexExpr = latexExpr.replaceAll("\\Tr",tr)
    latexExpr = latexExpr.replaceAll("{tr}","{Tr}")
    latexExpr = latexExpr.replaceAll("{TR}","{Tr}")
    latexExpr = latexExpr.replaceAll("\\re",re)
    latexExpr = latexExpr.replaceAll("\\Re",re)
    latexExpr = latexExpr.replaceAll("\\im",im)
    latexExpr = latexExpr.replaceAll("\\Im",im)
    latexExpr = latexExpr.replaceAll("<=","\\leq ")
    latexExpr = latexExpr.replaceAll(">=","\\geq ")
    latexExpr = latexExpr.replaceAll("!=","\\ne ")
    latexExpr = latexExpr.replaceAll("<"," < ")
    latexExpr = latexExpr.replaceAll(">"," > ")
    latexExpr = latexExpr.replaceAll("("," ( ")
    latexExpr = latexExpr.replaceAll(")"," ) ")
    latexExpr = latexExpr.replaceAll("["," [ ")
    latexExpr = latexExpr.replaceAll("]"," ] ")
    while(latexExpr.includes("  "))
        latexExpr = latexExpr.replaceAll("  "," ")
    return latexExpr
}

function extractLinks(input){
    const regex = /(?:U|P)(?:\^\\dagger)?_(\\(?:[^\\\s]+)|([^\\\s]))/g;
    let match;
    const results = [];

    while ((match = regex.exec(input)) !== null) {
      // match[0] is the full match, e.g. "U_\\alpha" or "U_b"
      // You can also check match[1] (the backslash case) or match[2] (the single letter case)
      results.push(match[0].replaceAll(/[{} ]/g,""));
    }

    return results;
}

function parseOneTerm(expression){
    
    if(expression==="")
        return ["",[],""]

    // Analyze the summation
    let regex = new RegExp(`\sum_{`, 'g');
    let matches = [...expression.matchAll(regex)];
    let sumStart = matches.map(match => match.index);

    function extractParenthesisContent(str, startPos, is_subscript=true) {

        if(is_subscript){
            let openCount = 0;
            let closeCount = 0;
            let content = '';
            let foundOpen = false;

            for (let i = startPos; i < str.length; i++) {
                    if (str[i] === "{") {
                        if (!foundOpen) {
                            foundOpen = true;
                        } else {
                            content += str[i];
                        }
                        openCount++;
                    } else if (str[i] === "}") {
                        closeCount++;
                        if (openCount === closeCount) {
                            if (str[i + 1] === '^') {
                                return content + "!" + (i + 1) + "!";
                            } else {
                                return content;
                            }
                        } else {
                            content += str[i];
                        }
                    } else if (foundOpen) {
                        content += str[i];
                    }
                }

                return null; // Return null if no completed parenthesis is found
        }else{
            let result = '';
            for (let j = startPos; j < str.length; j++) {
                if (str[j] === ' ' || str[j] === '\\') {
                    break;
                }
                result += str[j];
            }
            return result;
        }
    }

    let index_list = []
    let condition_dict = []
    let sumString = ""
    for(const ind of sumStart){
        let subscript = extractParenthesisContent(expression,ind)
        let supscript = "_"
        if(subscript.slice(-1)=="!"){
            let supind = parseInt(subscript.split("!")[1], 10)
            subscript = subscript.split("!")[0]

            if(expression[supind+1]=="\\"){
                supscript = "\\"+extractParenthesisContent(expression,supind+2,false)
            }else{
                supscript = expression[supind+1]
            }
        }
        let index = subscript.split(/(=|<|>|\\leq|\\geq|\\ne)/)[0]
        let condition_sym = subscript.split(/(=|<|>|\\leq|\\geq|\\ne)/)[1]
        let subject = subscript.split(/(=|<|>|\\leq|\\geq|\\ne)/)[2]

        index = index.replaceAll(" ","")
        condition_sym = condition_sym.replaceAll(" ","")
        subject = subject.replaceAll(" ","")
        
        sumString += "\\sum_{"+subscript+"}"
        if(supscript!="_")
            sumString+="^"+supscript

        // To sumarize, for each sum, we have the following 4 strings:
        // For \sum_{a=m}^n
        //  index           a
        //  condition_sym   =
        //  subject         m
        //  supscript       n  // is blank if there is no superscript

        index_list.push(index)
        condition_dict[index] = [condition_sym,subject,supscript]
    }

    let display = ""

    let links = extractLinks(expression)
    let nonsummed_links = []
    let nonsummed_polyakov = []
    let operator = ""
    if(expression.includes("\\text{Tr}"))
        operator+="\\tr~"
    for(const link of links){
        operator += link+" "
        let index = link.replaceAll("^\\dagger","").replaceAll("U_","").replaceAll("P_","")
        if(!index_list.includes(index)){
            if(link.includes("U")){
                nonsummed_links.push(link)
            }
            if(link.includes("P")){
                nonsummed_polyakov.push(link)
            }
        }
    }

    let pair_connection = []
    let linksplus = []
    for(const link of links){
        if(link.includes("P"))
            linksplus.push(link.replaceAll("P","U"))
        linksplus.push(link.replaceAll("P","U"))
    }
    if(links.length>1)
        linksplus.push(links[0].replaceAll("P","U"))
    for(let i=0;i<(linksplus.length-1);i++){
        pair_connection.push(linksplus[i]+linksplus[i+1])
    }

    function strMax(expression){
        if(expression.includes(","))
            return "\\text{max}("+expression.replace(" ","")+")"
        else
            return expression.replace(" ","")
    }
    function strMin(expression){
        if(expression.includes(","))
            return "\\text{min}("+expression.replace(" ","")+")"
        else
            return expression.replace(" ","")
    }
    operator = latexReformat(operator)
    display+="Relevant operator: $"+operator+"$<br>"
    display += "<ul>"
    for(const index of index_list){
        display += "<li>"
        let condition = condition_dict[index]
        if(condition[0]==="="){
            display += "$"+index+"$ running from $"+condition[1]+"$ to $"+condition[2]+"$"
        }else if(condition[0]==="<"){
            display += "$"+index+"$ running from $"+condition_dict[index_list[0]][1]+"$ to $"+strMin(condition[1])+"-1$"
        }else if(condition[0]==="\\leq"){
            display += "$"+index+"$ running from $"+condition_dict[index_list[0]][1]+"$ to $"+strMin(condition[1])+"$"
        }else if(condition[0]===">"){
            display += "$"+index+"$ running from $"+strMax(condition[1])+"+1$ to $"+condition_dict[index_list[0]][2]+"$"
        }else if(condition[0]==="\\geq"){
            display += "$"+index+"$ running from $"+strMax(condition[1])+"$ to $"+condition_dict[index_list[0]][2]+"$"
        }else if(condition[0]==="\\ne"){
            display += "$"+index+"$ running from $"+condition_dict[index_list[0]][1]+"$ to $"+condition_dict[index_list[0]][2]+"$ but $"+index+" \\ne "+condition[1]+"$"
        }
    }
    display += "</ul>"
    
    let numbered_links = []
    if(index_list.length>0){
        for(let mu=condition_dict[index_list[0]][1]; mu<=condition_dict[index_list[0]][2];mu++){
            numbered_links.push("U_"+mu)
            numbered_links.push("U^\\dagger_"+mu)
        }
    }
    for(const link of nonsummed_links){
        if(!numbered_links.includes(link))
            numbered_links.push(link)
    }
    for(const link of nonsummed_polyakov){
        if(!numbered_links.includes(link.replaceAll("P","U")))
            numbered_links.push(link.replaceAll("P","U"))
    }

    let numbered_links_count = []
    let polyakov_links_count = []
    let numbered_pairs = []
    for(const numbered_link of numbered_links){
        numbered_links_count[numbered_link] = 0
        polyakov_links_count[numbered_link] = 0
    }
    if(index_list.length>0){
        const resultList = iterateAll(index_list, condition_dict);

        for(const index of resultList){
            let replaced_operator = operator

            // before replacing, make sure it will not replace \dagger
            replaced_operator = replaced_operator.replaceAll("dagger","†")


            for(let axis=0; axis<index_list.length;axis++){
                replaced_operator = replaced_operator.replaceAll(index_list[axis],index[axis]+"")
            }
            replaced_operator = replaced_operator.replaceAll("†","dagger")
            //display += "$"+replaced_operator+"$<br>"
            for(const numbered_link of numbered_links){
                let numbered_polyakov = numbered_link.replaceAll("U","P")
                numbered_links_count[numbered_link] += (replaced_operator.split(numbered_link).length-1)
                polyakov_links_count[numbered_link] += (replaced_operator.split(numbered_polyakov).length-1)
            }

            for(const pair of pair_connection){
                let replaced_pair = pair
                for(let axis=0; axis<index_list.length;axis++){
                    replaced_pair = replaced_pair.replaceAll(index_list[axis],index[axis]+"")
                }
                numbered_pairs.push(replaced_pair)
            }
        }
    }else{
        for(const numbered_link of numbered_links){
            let numbered_polyakov = numbered_link.replaceAll("U","P")
            numbered_links_count[numbered_link] += (operator.split(numbered_link).length-1)
            polyakov_links_count[numbered_link] += (operator.split(numbered_polyakov).length-1)
        }

        for(const pair of pair_connection){
            numbered_pairs.push(pair)
        }
    }
    
    //for(const pair of numbered_pairs)
    //    display += "$"+pair + "$<br>"
        
    

    display += "Link variable counts per site: <br> "
    display += "&nbsp;&nbsp;&nbsp;&nbsp;"
    let linkCountStr = ""
    let ilink = 0
    let isNotClosedLoop = false
    for(const numbered_link of numbered_links){
        let count = numbered_links_count[numbered_link]+polyakov_links_count[numbered_link]
        if(count>0){
            if(ilink>0)
                linkCountStr += ", "
            linkCountStr += "$"+latexReformat(numbered_link)+"$&times;"+count
            ilink+=1
        }
        let u = ""
        let udagger = ""
        if(numbered_link.includes("\\dagger")){
            udagger = numbered_link
            u = udagger.replace("^\\dagger","")
            
        }else{
            u = numbered_link
            udagger = u.replace("_","^\\dagger_")
        }
        let nu = 0
        let nudagger = 0
        if(u in numbered_links_count)
            nu = numbered_links_count[u]
        if(udagger in numbered_links_count)
            nudagger = numbered_links_count[udagger]

        if(nu!=nudagger){
            isNotClosedLoop = true
        }
    }
    display += linkCountStr
    if(isNotClosedLoop)
        display += "<br><br><font color='red'>Warning: it seems the link variables in this term do not close into a loop. If this is intended, you can ignore this message.</font>"
    if(!expression.includes("\\text{Tr}"))
        display += "<br><br><font color='red'>Warning: the Wilson loop is not inside a trace. If this is intended, you can ignore this message.</font>"
    display += "<br><br>"

    return [display,
        numbered_links_count,
        polyakov_links_count,
        operator,
        sumString,
        numbered_pairs]
}

function iterateAll(indices, cond){
  let results = [];
  let current = {}; // object to store the current values for each key

  // first key should have "=" condition to define global bounds
  let firstKey = indices[0];
  if (cond[firstKey][0] !== "=") {
    throw new Error("The first index must have an '=' condition to define its range.");
  }
  const firstLow = cond[firstKey][1];
  const firstHigh = cond[firstKey][2];

  // iterate over the first index's range
  for (let v = firstLow; v <= firstHigh; v++) {
    current[firstKey] = v;
    // pass the first key's value as the global bound for subsequent keys
    iterate(1, indices, cond, current, results, firstHigh);
  }
  return results;
}

function int(x){
    return parseInt(x,10)
}

function notNumber(variable){
    return int(variable+"")+""==="NaN"
}

function iterate(indexIdx, indices, cond, current, results, globalBound) {
  // when all indices have been set, save a snapshot of the current values
  if (indexIdx === indices.length) {
    // Save the values in the order of the indices
    results.push(indices.map(key => current[key]));
    return;
  }

  const key = indices[indexIdx];
  const condition = cond[key];
  let low, high;

  function get_index(variable){
    if(int(variable+"")+""==="NaN"){
        return int(current[variable+""])
    }else{
        return int(variable+"")
    }
  }

  function get_min(variable){
    return Math.min(...variable.replace(" ","").split(',').map(get_index));
  }
  function get_max(variable){
    return Math.max(...variable.replace(" ","").split(',').map(get_index));
  }
  function is_equal(y,variable){
    return variable.split(',').map(get_index).includes(y);
  }

  if (condition[0] === "=") {
    // fixed range condition
    low = get_index(condition[1])
    high = get_index(condition[2])
    
  } else if (condition[0] === "<") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_min(dep))) return;
    low = get_index(0)
    high = get_min(dep)-1

  } else if (condition[0] === "\\leq") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_min(dep))) return;
    low = get_index(0)
    high = get_min(dep)

  } else if (condition[0] === ">") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_max(dep))) return;
    low = get_max(dep)+1
    high = get_index(globalBound)

  } else if (condition[0] === "\\geq") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_max(dep))) return;
    low = get_max(dep)
    high = get_index(globalBound)

  } else {
    low = get_index(0)
    high = get_index(globalBound)
  }

  // If the bounds are not valid, skip this branch.
  if (low > high) return;

  // iterate over the valid range for this key
  for (let v = low; v <= high; v++) {

    if (condition[0] === "\\ne"){
        const dep = condition[1]
        if(is_equal(v,dep))
            continue
    }

    current[key] = v;
    iterate(indexIdx + 1, indices, cond, current, results, globalBound);
  }
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//         Step 3
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
let group = 2
function createSUNDropdown() {
    ClearStep4();

    document.getElementById('step3title').innerHTML = "<br><hr><h2>Step 3: Character expansion</h2>"

    const container = document.getElementById("SUN");
    container.innerHTML = "Gauge group: "; // Clear previous content
    
    let selectedValue = "..."; // Default value
    
    const select = document.createElement("select");
    select.onchange = function(event) {
        selectedValue = event.target.value;
        if(selectedValue!="..."){
            group = parseInt(selectedValue.replaceAll("SU(","").replaceAll(")",""),10)
            console.log("Selected group:", selectedValue); // Logs updated value
            createCXForms()
        }else{
            const container = document.getElementById("characterExpansion");
            container.innerHTML = "";
        }
        
    };
    
    const optiondefault = document.createElement("option");
    optiondefault.value = `...`;
    optiondefault.textContent = `Select one`;
    select.appendChild(optiondefault);
    for (let i = 2; i <= 8; i++) {
        const option = document.createElement("option");
        option.value = `SU(${i})`;
        option.textContent = `SU(${i})`;
        select.appendChild(option);
    }
    
    container.appendChild(select);

}

let irrepData
function createCXForms() {

    let latexExpr = gbLatexExpr
    let parsed_operator = gbParsed_operator
    let parsed_sum = gbParsed_sum

    const container = document.getElementById("characterExpansion");
    container.innerHTML = ""; // Clear previous content
    let n = latexExpr.length
    irrepData = Array(n).fill(""); // Initialize array to store user input

    let term=1;
    for (const expression of latexExpr) {
        const form = document.createElement("form");
        const operator = parsed_operator[term-1]

        let display = ""

        display += ("<h4>$\\bullet$ "+ordinal(term)+" term "
            +"<font color='#888'>"
            +"::::::::::::::::::::::::::::::::::::"
            +"</font>"
            +"</h4>")

        let minusexpression = expression
        if(expression[0]=="-")
            minusexpression = expression.slice(1)
        else
            minusexpression = "-"+expression
        if(parsed_sum[term-1]!="")
            minusexpression = minusexpression.replaceAll(parsed_sum[term-1],"~")
        
        let placeholder1 = "0,0"
        let placeholder2 = "1,0"
        let placeholder3 = "1,1"

        if(group==2){
            placeholder1 = "[0]"
            placeholder2 = "[1]"
            placeholder3 = "[2]"
        }else{
            let nzeros = group-3
            for(let iz=0;iz<nzeros;iz++){
                placeholder1 += ",0"
                placeholder2 += ",0"
                placeholder3 += ",0"
            }
            placeholder1 = "["+placeholder1+"]"
            placeholder2 = "["+placeholder2+"]"
            placeholder3 = "["+placeholder3+"]"
        }
        let placeholder = placeholder1+", "+placeholder2+", "+placeholder3

        display += `
            &nbsp;&nbsp;&nbsp;&nbsp;
            &nbsp;&nbsp;&nbsp;&nbsp;
            $
            \\displaystyle e^{${minusexpression}}
            =\\sum_{r\\in\\text{Irreps}}f_r${operator.replaceAll("Tr}","Tr}_r")}$
            <br><br>
            &nbsp;&nbsp;&nbsp;&nbsp;
            &nbsp;&nbsp;&nbsp;&nbsp;
            <label>Expansion Irreps: </label>
            <input type="text" data-index="${term}" oninput="updateData(event)" value="${placeholder}">
            <div id="remarkCX${term}"></div>
        `;
        form.innerHTML = display;
        container.appendChild(form);
        term+=1;
    }
    container.innerHTML += "<br>"

    function toStep4(){
        
        document.querySelectorAll('input[data-index]').forEach(input => {
            const index = input.getAttribute("data-index");
            let input_data = input.value.replaceAll(" ","")

            let incorrectIrrep = false
            irrepData[index] = []
            for(const strTerm of input_data.split("],[")){
                let inStrTerm = strTerm.replaceAll("[","").replaceAll("]","")
                let intTerm = []
                for(const stri of inStrTerm.split(",")){
                    if(notNumber(stri)){
                        continue
                    }
                    intTerm.push(parseInt(stri,10))
                }
                if(intTerm.length!=group-1){
                    incorrectIrrep = true        
                    x = intTerm.length
                }
                irrepData[index].push(intTerm)
            }
            if(incorrectIrrep){
                irrepData[index] = []
                document.getElementById("remarkCX"+index).innerHTML = "<font color='red'>Irreps are not consistent with the group!</font>"
            }else{
                document.getElementById("remarkCX"+index).innerHTML = ""
            }
        });
        

        //proceed if there is no error
        let remarks = ""
        for(let index=1;index<=latexExpr.length;index++){
            remarks += document.getElementById("remarkCX"+index).innerHTML
        }
        if(remarks===""){
            linkIntegral()
            document.getElementById("Step5Button").innerHTML = `
            <button id="toStep5Button">To Step 5</button>
            <br>`
            document.getElementById("toStep5Button").onclick = function(){
                CGDecomposition();
            }
            MathJax.typesetPromise();
        }
        else{
            ClearStep4();
        }
        MathJax.typesetPromise()
    }

    window.updateData = function(event) {
        document.getElementById("Step4Button").innerHTML = `
        <button id="toStep4Button">To Step 4</button>
        <br>`
        document.getElementById("toStep4Button").onclick = function(){
            toStep4();
        }
        MathJax.typesetPromise();
    };
    document.getElementById("Step4Button").innerHTML = `
        <button id="toStep4Button">To Step 4</button>
        <br>`
    document.getElementById("toStep4Button").onclick = function(){
        toStep4();
    }
    MathJax.typesetPromise();
    
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//         Step 4
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
let integrand = {};
function linkIntegral(){
    ClearStep5();
    function swapKeys(obj) {
        let swapped = {};

        for (let key1 in obj) {
            for (let key2 in obj[key1]) {
                if (!swapped[key2]) {
                    swapped[key2] = {};
                }
                swapped[key2][key1] = obj[key1][key2];
            }
        }

        return swapped;
    }
    let display = ""
    integrand = swapKeys(gbLinkCount)

    display += `The information you gave is sufficient to determine the following:<br>
    <ul>
    <li>How many link variables there are in the group integrals.</li>
    <li>The representation sets for the group integrals (used as the indices of the vertex tensors).</li>
    <li>The connection of the matrix indices between the vertex tensors.</li>
    <li>And finally, the armillary sphere.</li>
    </ul>
    <br>
    The final step is to pass these information to the python script and let it handle the rest.
    <br><br>
    To summarize, here is all the information to be passed to the python script:
    <br><br>
    <pre><code>`
    let passed_param = ""

    passed_param += "irrep_set = []<br>"
    for(let term=1;term<irrepData.length;term++){
        let elem_str = "[ "
        let ielem = 0
        for(let elem of irrepData[term]){
            if(ielem>0)
                elem_str += ", "
            elem_str += `[${elem}]`
            ielem ++
        }
        elem_str += " ]"
        passed_param += `irrep_set.append( ${elem_str} ) # irrep set for term ${term-1}<br>`
    }

    passed_param += "<br><br>"

    passed_param += "axis_info = {}<br>"
    let first_key = true
    for(let key in integrand){
        let new_key = key.replaceAll("U^\\dagger","V").replaceAll("_","")
        passed_param += `axis_info["${new_key}"] = [`
        let j = 0
        for(let term in integrand[key]){
            for(let i=1; i<=integrand[key][term];i++){
                if(j>0)
                    passed_param += `, ${int(term)}`
                else
                    passed_param += `${int(term)}`
                j+=1
            }
        }
        passed_param += `]`
        if(first_key){
            passed_param += ` # a list of terms where each ${new_key} came from`
            first_key = false
        }
        passed_param += `<br>`
    }

    passed_param += "<br><br>"

    // reorganize the pair
    let pair_dict = {}
    for(let term in gbPairs){
        for(let pair of gbPairs[term]){
            pair = pair.replaceAll("U^\\dagger","V").replaceAll("_","")
            if(pair in pair_dict)
                pair_dict[pair].push(term)
            else
                pair_dict[pair] = [term]
        }
    }

    let ikey = 0
    passed_param += "connection_info = {}<br>"
    for(let pair in pair_dict){
        let temp = pair.replaceAll("U",",U").replaceAll("V",",V").split(",")
        let l1 = temp[1][0]
        let l2 = temp[2][0]
        let mu1 = temp[1].replaceAll("U","").replaceAll("V","")
        let mu2 = temp[2].replaceAll("U","").replaceAll("V","")

        let processed_pair = ""
        if(l1+l2=="UU")
            processed_pair = "-U"+mu1+" to "+"+U"+mu2
        else if(l1+l2=="UV")
            processed_pair = "-U"+mu1+" to "+"-V"+mu2
        else if(l1+l2=="VV")
            processed_pair = "+V"+mu1+" to "+"-V"+mu2
        else if(l1+l2=="VU")
            processed_pair = "+V"+mu1+" to "+"+U"+mu2
        else
            processed_pair = "??,"

        passed_param += `connection_info["${processed_pair}"] = [${pair_dict[pair]}]`

        if(ikey==0)
            passed_param += ` # a list of terms where`
        if(ikey==1)
            passed_param += ` # these connections came from.`
        passed_param += `<br>`
        ikey ++
    }

    display += passed_param+"</code></pre><br>"

    let title = "<br><hr><h2>Step 4: Summary</h2>"
    MathJax.typesetPromise();
    document.getElementById('output4').innerHTML = title+display
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//         Step 5
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
function CGDecomposition(){
    let display = ""

    let title = "<br><hr><h2>Step 5: Python Export</h2>"
    document.getElementById("output5").innerHTML = title+display

    MathJax.typesetPromise()
}