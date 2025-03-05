
function renderMath() {
    let input = document.getElementById("latexInput").value;
    input = latexReformat(input)
    if(input[0]=="+"){
        input = input.replace("+","")
    }
    document.getElementById("output1").innerHTML = "\\[" + input + "\\]";
    MathJax.typesetPromise();
    countUOccurrences();
}

function applyPresets() {
    let textArea = document.getElementById("latexInput");
    let checkboxes = document.querySelectorAll("input[type=checkbox]:checked");
    let presetText = "";
    
    checkboxes.forEach(box => {
        presetText += box.value + "\n";
    });
    
    textArea.value = presetText;
    adjustHeight(textArea);
}

function adjustHeight(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
}

function countUOccurrences() {

    let display = ""

    let latexExpr = document.getElementById("latexInput").value;
    latexExpr = latexReformat(latexExpr)

    latexExpr = latexExpr.split(/[\+\-]/)

    let term = 1
    for(const expression of latexExpr){
        let result = parseOneTerm(expression)
        if(result==="")
            continue
        //display += "<b>Term "+term+"</b>: \\["+expression+"\\]<br>"
        display += "<h4>$\\bullet$ Term "+term+"</h4>"
        display += result
        term+=1
    }

    document.getElementById("output2").innerHTML = display;
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
    return latexExpr
}

function parseOneTerm(expression){
    
    if(expression==="")
        return ""

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

    const resultList = iterateAll(index_list, condition_dict);

    let display = ""
    if(index_list.length==1)
        display += "There is 1 loop:"
    else
        display += "There are "+index_list.length+" loops:"
    display += "<ul>"
    for(const index of index_list){
        display += "<li>"
        let condition = condition_dict[index]
        if(condition[0]==="="){
            display += "$"+index+"$ running from $"+condition[1]+"$ to $"+condition[2]+"$"
        }else if(condition[0]==="<"){
            display += "$"+index+"$ running from $"+condition_dict[index_list[0]][1]+"$ to $"+condition[1]+"-1$"
        }else if(condition[0]==="\\leq"){
            display += "$"+index+"$ running from $"+condition_dict[index_list[0]][1]+"$ to $"+condition[1]+"$"
        }else if(condition[0]===">"){
            display += "$"+index+"$ running from $"+condition[1]+"+1$ to $"+condition_dict[index_list[0]][1]+"$"
        }else if(condition[0]==="\\geq"){
            display += "$"+index+"$ running from $"+condition[1]+"$ to $"+condition_dict[index_list[0]][2]+"$"
        }else if(condition[0]==="\\ne"){
            display += "$"+index+"$ running from $"+condition_dict[index_list[0]][1]+"$ to $"+condition_dict[index_list[0]][2]+"$ but $"+index+" \\ne "+condition[1]+"$"
        }
    }
    display += "</ul>"
    /*
    display += "Index values:<br>"
    for(const index of resultList){
        display += index+"<br>"
    }
    */
    return display
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

  if (condition[0] === "=") {
    // fixed range condition
    low = condition[1];
    high = condition[2];
  } else if (condition[0] === "<") {
    const dep = condition[1];
    if (current[dep] === undefined) return;
    low = 1;
    high = parseInt(current[dep],10) - 1;
  } else if (condition[0] === "\\leq") {
    const dep = condition[1];
    if (current[dep] === undefined) return;
    low = 1;
    high = parseInt(current[dep],10);
  } else if (condition[0] === ">") {
    const dep = condition[1];
    if (current[dep] === undefined) return;
    low = parseInt(current[dep],10) + 1;
    high = globalBound;
  } else if (condition[0] === "\\geq") {
    const dep = condition[1];
    if (current[dep] === undefined) return;
    low = parseInt(current[dep],10);
    high = globalBound;
  } else {
    low = 1;
    high = globalBound;
  }

  // If the bounds are not valid, skip this branch.
  if (low > high) return;

  // iterate over the valid range for this key
  for (let v = low; v <= high; v++) {

    if (condition[0] === "\\ne"){
        const dep = condition[1]
        if(v==parseInt(current[dep],10))
            continue
    }

    current[key] = v;
    iterate(indexIdx + 1, indices, cond, current, results, globalBound);
  }
}
