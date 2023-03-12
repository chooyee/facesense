
var mandatoryElements = document.querySelectorAll('.mandatory')
mandatoryElements.forEach(element => {
    element.addEventListener('input',(event) => {
    //console.log(element)
    element.classList.remove('is-invalid');

    });
})        

MandatoryValidation = ()=>{
    let submit = true
    mandatoryElements.forEach(element => {
        if (element.value==='')
        {
            element.classList.add('is-invalid')
            submit = false
        }        
    });
    return submit
}
