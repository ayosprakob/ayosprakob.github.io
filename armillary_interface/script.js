
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Global variables :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

const irrepSetsDiv = document.getElementById('irrep-sets');
const termsDiv = document.getElementById('terms');
const section2 = document.getElementById('section2');
const section3 = document.getElementById('section3');

const dropdownN = document.getElementById('dropdown-n');
const dropdownDimensions = document.getElementById('dropdown-dimensions');

var num_sets = 0;


// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Section 1 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

// Gauge group ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

dropdownN.addEventListener('change', () => {

    if (irrepSetsDiv.children.length == 0 || confirm(`Changing N will reset everything. Proceed?`)) {
        irrepSetsDiv.innerHTML = '';
        termsDiv.innerHTML = '';
        section2.style.display = 'none';
    } else {
        dropdownN.value = dropdownN.dataset.prev || 2;
    }
    var trv = "0";
    var fnd = "1";
    for(let i=0;i<dropdownN.value-2;i++){
        trv += ",0";
        fnd += ",0";
    }
    trv = "("+trv+")";
    fnd = "("+fnd+")";
    var fermionic_set = "{"+trv+", "+fnd+"}";
    document.getElementById("fermionic-set-template").innerHTML = fermionic_set;
});
dropdownN.dataset.prev = dropdownN.value;


// Irreps set :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

document.getElementById('add-irrep-set').addEventListener('click', () => {
    const irrepSet = document.createElement('div');
    const inputId = `irrep-name-${irrepSetsDiv.children.length + 1}`;
    var TemplateSetVal = ``
    for (let i=1; i<=3; i++){
        if(i>1){
            TemplateSetVal += ", "
        }
        TemplateSetVal += `(${i-1}`
        for (let r=1; r<=dropdownN.value-2; r++){
            TemplateSetVal += ",0"
        }
        TemplateSetVal += ")"
    }
    num_sets += 1
    irrepSet.innerHTML = `
    <label>Set ${num_sets}:
    <input type="text" id="${inputId}" class="irrep-name"
    value="${TemplateSetVal}"
    >
    </label>
    <button class="remove-irrep">Remove</button>
    `;
    irrepSetsDiv.appendChild(irrepSet);

    irrepSet.querySelector('.remove-irrep').addEventListener('click', () => {
        irrepSet.remove();
        updateIrrepDropdowns();
        if (irrepSetsDiv.children.length === 0) section2.style.display = 'none';
    });

    irrepSet.querySelector('.irrep-name').addEventListener('change', () => {
        updateIrrepDropdowns();
    });

    updateIrrepDropdowns();

    
});

document.getElementById('to-section2').addEventListener('click', () => {
    if (irrepSetsDiv.children.length > 0){section2.style.display = 'block';}
    else{alert("Add the set of irreps first!");}
});

// Update irreps after each modification ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function updateIrrepDropdowns() {
    const irrepNames = Array.from(irrepSetsDiv.children).map(irrepSet => {
        const nameInput = irrepSet.querySelector('.irrep-name');
        return nameInput.value || '(Unnamed)';
    });

    const irrepOptions = irrepNames.map(name => `<option value="${name}">${name}</option>`).join('');

    Array.from(termsDiv.querySelectorAll('.irrep-ref')).forEach(select => {
        const currentValue = select.value;
        select.innerHTML = irrepOptions;
        if (irrepNames.includes(currentValue)) {
            select.value = currentValue;
        }
    });
}

updateIrrepDropdowns();


// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Section 2 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

// Lattice dimensions :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

dropdownDimensions.addEventListener('change', () => {
    if (termsDiv.children.length == 0 || confirm('Changing dimensions will reset all operators. Proceed?')) {
        termsDiv.innerHTML = '';
    } else {
        dropdownDimensions.value = dropdownDimensions.dataset.prev || 2;
    }
});
dropdownDimensions.dataset.prev = dropdownDimensions.value;


var elements = document.getElementsByClassName("update-dropdowns");
for (let i=0;i<elements.length;i++){
    elements[i].addEventListener('click', updateIrrepDropdowns);
}

termsDiv.addEventListener('input', event => {
    if (event.target.classList.contains('irrep-name')) {
        updateIrrepDropdowns();
    }
});

// Spoiler
function add_hint(hint_name){
    document.getElementById(hint_name+"_button").addEventListener("click", function() {
        let wrapper = document.getElementById(hint_name);
        if (wrapper.classList.contains("show")) {
          wrapper.classList.remove("show");
          this.textContent = "Show input hint";
        } else {
          wrapper.classList.add("show");
          this.textContent = "Hide input hint";
        }
      });
    const style = document.createElement("style");
    style.textContent = `
        #${hint_name} {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-in-out;
        }
        #${hint_name}.show {
            max-height: 1000px;
        }
    `;
    document.head.appendChild(style);

    const spoiler = document.getElementById(hint_name);

    function toggleSpoiler() {
        if (spoiler.style.maxHeight && spoiler.style.maxHeight !== "0px") {
            // Collapse
            spoiler.style.maxHeight = spoiler.scrollHeight + "px"; // Set to current height first
            setTimeout(() => {
                spoiler.style.maxHeight = "0px"; // Then collapse
            }, 10); // Small delay to trigger transition
        } else {
            // Expand
            spoiler.style.maxHeight = spoiler.scrollHeight + "px";
        }
    }

    // Example: Toggle on button click
    document.getElementById(hint_name+"_button").addEventListener("click", toggleSpoiler);

}
add_hint("spoiler1")
add_hint("spoiler2")

document.getElementById('add-term').addEventListener('click', () => add_term("") );
document.getElementById('add-term-plaquette').addEventListener('click', () => add_term("plaquette") );
document.getElementById('add-term-theta').addEventListener('click', () => add_term("theta") );
document.getElementById('add-term-polyakov').addEventListener('click', () => add_term("polyakov") );

function add_term(template){
    const dimensions = parseInt(dropdownDimensions.value, 10);
    const termDiv = document.createElement('div');

    var skip_term = true

    var term_name = ""
    var term_tex = ""
    if (template==""){
        skip_term = false
    }
    if (template=="plaquette"){
        term_name = "Plaquette action"
        term_tex = `U_\\mu U_\\nu U'_\\mu U'_\\nu`
        skip_term = false
    }
    if (template=="theta"){
        term_name = "Theta term"
        if(dimensions==2){
            alert(`Theta term in 1+1D usually can be grouped with the plaquette action. If you already have the plaquette action, I recommend removing the theta term.`)
            term_tex = `U_\\mu U_\\nu U'_\\mu U'_\\nu`
            skip_term = false
        }else if(dimensions==4){
            term_tex = `U_\\mu U_\\nu U'_\\mu U'_\\nu U_\\rho U_\\sigma U'_\\rho U'_\\sigma`
            skip_term = false
        }else{
            alert("")
        }
    }
    if (template=="polyakov"){
        term_name = "Polyakov loop"
        term_tex = `L_\\mu`
        skip_term = false
    }

    if(skip_term){
        return 1
    }

    termDiv.classList.add('term');
    termDiv.innerHTML = `
        <br>
        <div class="term-tex-out"></div>
        <div>
        Term Name: <input type="text" value="${term_name}" class="term-name">
        <button class="remove-term">Remove</button>
        <br>
        Operator: <input type="text" value="${term_tex}" class="term-tex">
        <button class="render-tex">Render</button>
        <br>
        <div class="sum-rule">
        </div>
        <br>
        Expansion set: 
        <select class="irrep-ref">
            ${Array.from(irrepSetsDiv.children).map((irrepSet, index) => {
                const nameInput = irrepSet.querySelector('.irrep-name');
                const setName = `Set ${index + 1}`;
                return `<option value="${nameInput.value}">${setName}</option>`;
            }).join('')}
        </select>
        <input type="text" value="" class="link-information" style="display: none;">
        
    `;

    termDiv.querySelector('.remove-term').addEventListener('click', () => {
        termDiv.remove();
    });

    termDiv.querySelector('.render-tex').addEventListener('click', () => {
        var display = ""
        var error = ""
        var texContent = "$$"+termDiv.querySelector('.term-tex').value+"$$"
        texContent = texContent.replace(/'/g,"^\\dagger")

        // Begin Error detection :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        var indices = UString(termDiv.querySelector('.term-tex').value)
        if(indices[0]=="error") error = "Invalid syntax.";
        // check if it closes into a loop
        var is_loop = true;
        for(let i in indices){
            var index = indices[i]
            var conj
            if(index.substring(0, 2)=="+u") conj=index.replace("+u","-u");
            else if(index.substring(0, 2)=="-u") conj=index.replace("-u","+u");
            else continue;
            const countA = indices.filter(item => item === index).length;
            const countB = indices.filter(item => item === conj).length;
            if(countA!=countB){
                is_loop = false;
                break;
            }
        }
        if(!is_loop) error = "The operator does not form a loop.";

        // End Error detection :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        if(indices[0]=="error"){
            display = `<font color='red'>Error: ${error}</font>`
            termDiv.querySelector('.link-information').value = "error"
        }else{
            if(texContent.length>0){
                display = texContent;
                termDiv.querySelector('.link-information').value = indices
            }
            if(error!=""){
                display += `<br><br><font color='red'>Error: ${error}</font>`
                termDiv.querySelector('.link-information').value = "error"
            }
        }

        termDiv.querySelector('.term-tex-out').innerHTML = display

        // Sum rules
        if(error==""){
            var sumRuleDiv = termDiv.querySelector('.sum-rule')
            var unique_indices = [];
            var sum_rule_val = ""
            for(let i in indices){
                var index = indices[i].slice(2)
                var texIndex = "$"+mapTeXExpression(index)+"$"
                if(!unique_indices.includes(texIndex))
                    unique_indices.push(texIndex)
            }
            for(let i in unique_indices){
                var index = unique_indices[i]
                sum_rule_val += `<div>`
                sum_rule_val += `&bullet; ${index} sums over `
                sum_rule_val += `<select class="sum-rule-option">`
                sum_rule_val += `<option value="custom">range</option>`;
                var temp = 1
                for(let j in unique_indices){
                    if(j==i) continue;
                    var index2 = unique_indices[j]
                    sum_rule_val += `<option value="${index}<${index2}">${index}<${index2}</option>`
                    sum_rule_val += `<option value="${index}&leq;${index2}">${index}&leq;${index2}</option>`
                    sum_rule_val += `<option value="${index}>${index2}">${index}>${index2}</option>`
                    sum_rule_val += `<option value="${index}&geq;${index2}">${index}&geq;${index2}</option>`
                    sum_rule_val += `<option value="${index}&ne;${index2}">${index}&ne;${index2}</option>`
                    temp+=1
                }
                sum_rule_val += `</select>`
                sum_rule_val += `</div>`

            }
            // Insert the generated HTML
            sumRuleDiv.innerHTML = sum_rule_val

            // Add event listeners to the dropdowns
            document.querySelectorAll('.sum-rule-option').forEach(select => {
                select.addEventListener('change', function () {
                    let parentDiv = this.parentElement;
                    let existingInput = parentDiv.querySelector('.custom-input');

                    if (this.value === "custom") {
                        if (!existingInput) {
                            let input = document.createElement('input');
                            input.type = "text";
                            input.classList.add('custom-input');

                            var input_value = []
                            for(let d=1;d<=dimensions;d++)
                                input_value.push(d)

                            input.value = input_value;
                            parentDiv.appendChild(input);
                        }
                    } else {
                        if (existingInput) {
                            existingInput.remove();
                        }
                    }
                });
                var event = new Event('change');
                select.dispatchEvent(event);
            });
        }else{
            termDiv.querySelector('.sum-rule').innerHTML = ""
        }

        MathJax.typesetPromise();
    });

    termsDiv.appendChild(termDiv);

    termDiv.querySelector('.render-tex').click()

    updateIrrepDropdowns();
}

function UString(str) {
    str = mapGreekExpression(str).replace(/ /g,"");

    const pattern = /^([U|L]'?_[a-zα-ω])+$/;

    if (!pattern.test(str)) {
        return ["error"];
    }

    return str.match(/[U|L]'?_([a-zα-ω])/g).map(match => {
        let char = match.match(/([a-zα-ω])/)[1];
        
        if (match.startsWith("U'_")) {
            return `-u${char}`;  // For U'_a return -ua
        } else if (match.startsWith("U_")) {
            return `+u${char}`;  // For U_a return +ua
        } else if (match.startsWith("L'_")) {
            return `-l${char}`;  // For L'_a return -la
        } else if (match.startsWith("L_")) {
            return `+l${char}`;  // For L_a return +la
        }
    });
}

function mapGreekExpression(expression) {
  const greekMap = {
    '\\alpha': 'α', '\\Alpha': 'Α',
    '\\beta': 'β', '\\Beta': 'Β',
    '\\gamma': 'γ', '\\Gamma': 'Γ',
    '\\delta': 'δ', '\\Delta': 'Δ',
    '\\epsilon': 'ε', '\\Epsilon': 'Ε',
    '\\zeta': 'ζ', '\\Zeta': 'Ζ',
    '\\eta': 'η', '\\Eta': 'Η',
    '\\theta': 'θ', '\\Theta': 'Θ',
    '\\iota': 'ι', '\\Iota': 'Ι',
    '\\kappa': 'κ', '\\Kappa': 'Κ',
    '\\lambda': 'λ', '\\Lambda': 'Λ',
    '\\mu': 'μ', '\\Mu': 'Μ',
    '\\nu': 'ν', '\\Nu': 'Ν',
    '\\xi': 'ξ', '\\Xi': 'Ξ',
    '\\omicron': 'ο', '\\Omicron': 'Ο',
    '\\pi': 'π', '\\Pi': 'Π',
    '\\rho': 'ρ', '\\Rho': 'Ρ',
    '\\sigma': 'σ', '\\Sigma': 'Σ',
    '\\tau': 'τ', '\\Tau': 'Τ',
    '\\upsilon': 'υ', '\\Upsilon': 'Υ',
    '\\phi': 'φ', '\\Phi': 'Φ',
    '\\chi': 'χ', '\\Chi': 'Χ',
    '\\psi': 'ψ', '\\Psi': 'Ψ',
    '\\omega': 'ω', '\\Omega': 'Ω'
  };
  
  return expression.replace(/\\[a-zA-Z]+/g, match => greekMap[match] || match);
}

function mapTeXExpression(expression) {
  const reverseGreekMap = {
        'α': '\\alpha', 'Α': '\\Alpha',
        'β': '\\beta', 'Β': '\\Beta',
        'γ': '\\gamma', 'Γ': '\\Gamma',
        'δ': '\\delta', 'Δ': '\\Delta',
        'ε': '\\epsilon', 'Ε': '\\Epsilon',
        'ζ': '\\zeta', 'Ζ': '\\Zeta',
        'η': '\\eta', 'Η': '\\Eta',
        'θ': '\\theta', 'Θ': '\\Theta',
        'ι': '\\iota', 'Ι': '\\Iota',
        'κ': '\\kappa', 'Κ': '\\Kappa',
        'λ': '\\lambda', 'Λ': '\\Lambda',
        'μ': '\\mu', 'Μ': '\\Mu',
        'ν': '\\nu', 'Ν': '\\Nu',
        'ξ': '\\xi', 'Ξ': '\\Xi',
        'ο': '\\omicron', 'Ο': '\\Omicron',
        'π': '\\pi', 'Π': '\\Pi',
        'ρ': '\\rho', 'Ρ': '\\Rho',
        'σ': '\\sigma', 'Σ': '\\Sigma',
        'τ': '\\tau', 'Τ': '\\Tau',
        'υ': '\\upsilon', 'Υ': '\\Upsilon',
        'φ': '\\phi', 'Φ': '\\Phi',
        'χ': '\\chi', 'Χ': '\\Chi',
        'ψ': '\\psi', 'Ψ': '\\Psi',
        'ω': '\\omega', 'Ω': '\\Omega'
    };
  
  return expression.replace(/\\[a-zA-Z]+/g, match => reverseGreekMap[match] || match);
}

// This is for initial debugging
function display_linkdata(){

    section3.style.display = 'block';

    const output = {
        N: parseInt(dropdownN.value),
        dim: parseInt(dropdownDimensions.value),
        terms: []
    };

    let has_error = false;
    Array.from(termsDiv.children).forEach(term => {
        const term_name = term.querySelector('.term-name').value;
        const irrep_set = term.querySelector('.irrep-ref').value;
        const link_info = term.querySelector('.link-information').value;

        var sumRulesData = [];
        term.querySelectorAll('.sum-rule').forEach(sumRuleDiv => {
            let ruleEntries = [];

            sumRuleDiv.querySelectorAll('.sum-rule-option').forEach(select => {
                let selectedValue = select.value;

                if (selectedValue === "custom") {
                    let input = select.parentElement.querySelector('.custom-input');
                    ruleEntries.push(`custom:${input ? input.value : ''}`);
                } else {
                    ruleEntries.push(selectedValue.replace(/\$/g,""));
                }
            });

            sumRulesData.push(ruleEntries);
        });

        output.terms.push({ term_name, irrep_set, link_info,sumRulesData});
        if (link_info === "error") has_error = true;
    });

    if(has_error){
        alert("You still have some error in step 2!");
    }
    document.getElementById('output').textContent = JSON.stringify(output, null, 2);
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Section 2 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

document.getElementById('to-section3').addEventListener('click', () => display_linkdata());
