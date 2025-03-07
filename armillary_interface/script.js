
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

function countUOccurrences() {
    document.getElementById("output2").innerHTML = "";
    document.getElementById("interlude1to2").innerHTML = "";

    let display = ""

    let latexExpr = document.getElementById("latexInput").value;
    latexExpr = latexReformat(latexExpr)

    latexExpr = latexExpr.split(/[\+\-]/)

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

    let term = 1
    let linkCount = []
    let parsed_operator = []
    for(const expression of latexExpr){
        let result = parseOneTerm(expression)
        for(let key in result[1]){
            if(key in linkCount)
                linkCount[key] += result[1][key]
            else
                linkCount[key] = result[1][key]
            
        }
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
        parsed_operator.push(result[2])
        term+=1
    }
    if(display!=""){
        document.getElementById("interlude1to2").innerHTML = "<center><img src='images/downarrow.jpg' width='10%'></center>";
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

function extractLinks(input) {
    const regex = /U(?:\^\\dagger)?_(\\(?:[^\\\s]+)|([^\\\s]))/g;
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
    let operator = ""
    if(expression.includes("\\text{Tr}"))
        operator+="\\tr~"
    for(const link of links){
        operator += link+" "
        let index = link.replaceAll("^\\dagger","").replaceAll("U_","")
        if(!index_list.includes(index))
            nonsummed_links.push(link)
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
    for(const link of nonsummed_links)
        if(!numbered_links.includes(link))
            numbered_links.push(link)

    let numbered_links_count = []
    for(const numbered_link of numbered_links){
        numbered_links_count[numbered_link] = 0
        //display+=numbered_link+"<br>"
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
                numbered_links_count[numbered_link] += (replaced_operator.split(numbered_link).length-1)
            }
        }
    }else{
        for(const numbered_link of numbered_links)
            numbered_links_count[numbered_link] += (operator.split(numbered_link).length-1)
    }
    
        
    

    display += "Link variable counts per site: <br> "
    display += "&nbsp;&nbsp;&nbsp;&nbsp;"
    let linkCountStr = ""
    let ilink = 0
    let isNotClosedLoop = false
    for(const numbered_link of numbered_links){
        let count = numbered_links_count[numbered_link]
        if(count>0){
            if(ilink>0)
                linkCountStr += ", "
            linkCountStr += "$"+latexReformat(numbered_link)+"$&times;"+count
            ilink+=1
        }
        if(numbered_link.includes("\\dagger")){
            let udagger = numbered_link
            let u = udagger.replace("^\\dagger","")
            if(numbered_links_count[u]!=numbered_links_count[udagger])
                isNotClosedLoop = true
        }
    }
    display += linkCountStr
    if(isNotClosedLoop)
        display += "<br><br><font color='red'>Warning: it seems the link variables in this term do not close into a loop. If this is intended, you can ignore this message.</font>"
    if(!expression.includes("\\text{Tr}"))
        display += "<br><br><font color='red'>Warning: the Wilson loop is not inside a trace. If this is intended, you can ignore this message.</font>"
    display += "<br><br>"

    return [display,numbered_links_count,operator+";"+linkCountStr]
}

function iterateAll(indices, cond) {
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

    function int(x){
        return parseInt(x,10)
    }

  function notNumber(variable){
    return int(variable+"")+""==="NaN"
  }

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
