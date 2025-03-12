/** :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
/**      Place this at the beginning of your script. DO NOT MODIFY.                        */
/** :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
/**/ 
/**/ let python_result = ""; // Global variable for Python output
/**/ let pyodideInstance = null; // Pyodide instance placeholder
/**/ let pyodideReady = false; // Flag to track initialization status
/**/ 
/**/ async function initializePyodide() {
/**/     if (!pyodideInstance) {
/**/         console.log("Initializing Pyodide...");
/**/         pyodideInstance = await loadPyodide({
/**/             indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.3/full/"
/**/         });
/**/ 
/**/         await pyodideInstance.loadPackage("numpy");
/**/ 
/**/         pyodideInstance.globals.set("print", (text) => {
/**/             python_result += ""+text + "<br>";
/**/         });
/**/         pyodideReady = true;
/**/     }
/**/ 
/**/     while (!pyodideReady) {
/**/         await new Promise(resolve => setTimeout(resolve, 10));
/**/         // Wait 10ms before checking again
/**/     }
/**/ }
/**/ 
/**/ // Run Python function
/**/ async function run_python(args, script) {
/**/     await initializePyodide(); // Make sure Pyodide is fully initialized
/**/     python_result = ""
/**/     let pythonCode = script;
/**/ 
/**/     //injecting the jsprint definition into the script
/**/     pythonCode = pythonCode.replace(/(^|[\s;\t])print/g, '$1jsprint');
/**/     let jsprint = `
/**/     def jsprint(*args,end="\\n"):
/**/      ret = ""
/**/      for i,elem in enumerate(args):
/**/       ret += str(elem).replace("\\n","<br>")
/**/       if i!=len(args) : ret += " "
/**/       else: ret += end
/**/      print(ret)
/**/     `.replaceAll(`/**/     `,'')
/**/     pythonCode = jsprint+pythonCode
/**/ 
/**/     for (let key in args) {
/**/         pythonCode = pythonCode.replaceAll(`__${key}__`, args[key]);
/**/     }
/**/     try {
/**/         await pyodideInstance.runPythonAsync(pythonCode);
/**/     } catch (err) {
/**/         python_result = 'Error: ' + err;
/**/     }
/**/ }
/**/ 
/**/ // Automatically initialize Pyodide on script load
/**/ initializePyodide();
/**/ 
/**/ // Expose `run_python` globally so it can be called from anywhere
/**/ window.run_python = run_python;
/**/ 
/** :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
/**      Place this at the beginning of your script. DO NOT MODIFY.                        */
/** :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: */

// This is the example where you run the python script inside a function
async function my_function(){
    
    document.getElementById("output").innerHTML = "running the script...";

    await run_python({"matrix_input":`np.random.rand(3,3)`},SVD_script)
    //SVD_script is the script imported from py_script.js

    //print the result to the prepared div tag
    document.getElementById("output").innerHTML = python_result;
    
}

// This is the example where you run the python script via an event listener
function update_button(){

    document.addEventListener('click', async () => {
    if (event.target && event.target.id === 'button-id') {
        document.getElementById("output").innerHTML = "running the script...";

        await run_python({"matrix_input":`np.random.rand(3,3)`},SVD_script)
        //SVD_script is the script imported from py_script.js

        //print the result to the prepared div tag
        document.getElementById("output").innerHTML = python_result;

    }})
    
}