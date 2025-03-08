
function renderMath() {
    let input = document.getElementById("latexInput").value;
    if(input.length>0){
        input = latexReformat(input)
        if(input[0]=="+"){
            input = input.replace("+","")
        }
        document.getElementById("output1").innerHTML = "<div class=\"wrap\">\\[" + input + "\\]</div>";
        MathJax.typesetPromise();
        countUOccurrences();
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

let gbLatexExpr
let gbParsed_operator
let gbParsed_sum
let gbLinkCount
function countUOccurrences() {
    document.getElementById("output2").innerHTML = "";
    document.getElementById("interlude1to2").innerHTML = "";
    document.getElementById("SUN").innerHTML = "";
    document.getElementById("characterExpansion").innerHTML = "";
    document.getElementById("output4").innerHTML = "";

    let display = "Please confirm that everything is correct. If not, check your input again."

    let latexExpr = document.getElementById("latexInput").value;
    latexExpr = latexReformat(latexExpr)
    latexExpr = latexExpr.replaceAll("-","+-")
    latexExpr = latexExpr.split("+").slice(1)

    

    let term = 1
    let linkCount = []
    let parsed_operator = []
    let parsed_sum = []
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
        term+=1
    }
    if(display!=""){
        document.getElementById("interlude1to2").innerHTML = "<center><img src='images/downarrow.jpg' width='10%'></center>";
        document.getElementById("interlude2to3").innerHTML = "<center><img src='images/downarrow.jpg' width='10%'></center>";
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

    let notice_text = ""//"<i><font color='blue'>Note: If the result looks strange, the reason might be that the input is ambiguous.</font></i><nl>"
    document.getElementById("output2").innerHTML = notice_text+display;
    MathJax.typesetPromise();

    gbLatexExpr = latexExpr
    gbParsed_operator = parsed_operator
    gbParsed_sum = parsed_sum
    gbLinkCount = linkCountByTerm

    createSUNDropdown()
}

function latexReformat(latexExpr){
    latexExpr = latexExpr.replaceAll("\\tr","\\text{Tr}")
    latexExpr = latexExpr.replaceAll("\\Tr","\\text{Tr}")
    latexExpr = latexExpr.replaceAll("\\re","\\text{Re}")
    latexExpr = latexExpr.replaceAll("\\Re","\\text{Re}")
    latexExpr = latexExpr.replaceAll("\\im","\\text{Im}")
    latexExpr = latexExpr.replaceAll("\\Im","\\text{Im}")
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

    return [display,numbered_links_count,polyakov_links_count,operator,sumString]
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
    low = get_index(1)
    high = get_min(dep)-1

  } else if (condition[0] === "\\leq") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_min(dep))) return;
    low = get_index(1)
    high = get_min(dep)

  } else if (condition[0] === ">") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_max(dep))) return;
    low = get_max(dep)+1
    high = get_index(globalBound)

  } else if (condition[0] === "\\geq") {
    const dep = condition[1];
    if (current[dep] === undefined && notNumber(get_max(dep))) return;
    //low = get_index(dep)
    low = get_max(dep)
    high = get_index(globalBound)

  } else {
    low = get_index(1)
    high = get_index(globalBound)
  }

  // If the bounds are not valid, skip this branch.
  if (low > high) return;

  // iterate over the valid range for this key
  for (let v = low; v <= high; v++) {

    if (condition[0] === "\\ne"){
        const dep = condition[1]
        //if(v==get_index(dep))
        if(is_equal(v,dep))
            continue
    }

    current[key] = v;
    iterate(indexIdx + 1, indices, cond, current, results, globalBound);
  }
}

let group = 2
function createSUNDropdown() {
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
            =\\sum_{r\\in\\text{Irreps}}f_r${operator}$
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

    window.updateData = function(event) {
        const index = event.target.getAttribute("data-index");
        let input_data = event.target.value.replaceAll(" ","")

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

        //proceed if there is no error
        let remarks = ""
        for(let index=1;index<=latexExpr.length;index++){
            remarks += document.getElementById("remarkCX"+index).innerHTML
        }
        if(remarks==="")
            linkIntegral()
        else
            document.getElementById('output4').innerHTML = ""
    };

    //proceed if there is no error
    let remarks = ""
    for(let index=1;index<=latexExpr.length;index++){
        remarks += document.getElementById("remarkCX"+index).innerHTML
    }
    if(remarks===""){
        document.getElementById("interlude3to4").innerHTML = "<center><img src='images/downarrow.jpg' width='10%'></center>";
        linkIntegral()
    }
    else
        document.getElementById('output4').innerHTML = ""
    MathJax.typesetPromise()
}

function linkIntegral(){
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
    let integrand = swapKeys(gbLinkCount)

    display += "<ul>"
    for(let key in integrand){
        if(key.includes("dagger")){

            display += "<li>"
            let Udag = key
            let U = key.replace("^\\dagger","")
            let I = U.replace("U","I")
            let strIntegral = I+"=\\int d"+U+"~"
            for(let term in integrand[U]){
                let nU = integrand[U][term]
                let termname = ordinal(int(term)+1)
                strIntegral += "\\underset{\\text{"+termname+" term}}{\\underbrace{"
                for(let i=0;i<nU;i++)
                    strIntegral += U
                strIntegral += "}}~"
            }
            for(let term in integrand[Udag]){
                let nUdag = integrand[Udag][term]
                let termname = ordinal(int(term)+1)
                strIntegral += "\\underset{\\text{"+termname+" term}}{\\underbrace{"
                for(let i=0;i<nUdag;i++)
                    strIntegral += Udag
                strIntegral += "}}~"
            }
            display += "$\\displaystyle "+strIntegral+"$<br><br>"
            display += "</li>"
        }
    }
    display += "</ul>"

    document.getElementById('output4').innerHTML = display
}
