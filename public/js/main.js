const btnUpload = document.getElementById('btnUpload')
const txtEmail = document.getElementById('txtEmail')
const formElements = document.querySelectorAll('.form-control')
const divResult = document.getElementById('divResult')

document.addEventListener('DOMContentLoaded', function () {    
    
}, false
);

function checkStatus(response) {
            
    if (response.status >= 200 && response.status < 300) {
        return Promise.resolve(response)
    } else {                
        return Promise.reject(response.text())
    }
}

function parseJson(response) {
    return response.json()
}

function fetchData(req) {
    // var req = new Request('https://api.acme.com/users.json', {
    //     method: 'post',
    //     mode: 'cors',
    //     redirect: 'follow'
    //     headers: {
    //     "Content-type": "application/x-www-form-urlencoded; charset=UTF-8"
    //     },
    //     body: 'foo=bar&lorem=ipsum'
    // });
    return fetch(req)
        .then(checkStatus)
        .then(parseJson)
        .then(function (data) {
            console.log('Request succeeded with JSON response', data);
            return data
        }).catch(async function (error) {
            errMsg =  await error
            console.log('Request failed', errMsg);
            divResult.appendChild(CreateErrorMsgCard(errMsg))
            btnUpload.disabled = false
        });
}
btnUpload.addEventListener('click', async (e)=>{
    btnUpload.disabled = true
    var submit = MandatoryValidation()
    if (customFile.files.length < 1)
    {
        alert("Please select a file to upload!")
    }
    if (submit)
    {
        // formElements.forEach(element => {   
        //     let id = element.id
        //     data[id] = element.value        
        // });

        let formData = new FormData(); 
        formData.append("id", txtEmail.value);
        formData.append("file", customFile.files[0]);
        var req = new Request('face/v1/detect', {
            method: 'post',
            mode: 'cors',
            redirect: 'follow',          
            body: formData
        });
        result = await fetchData(req)
        btnUpload.disabled = false
        divResult.appendChild(CreateCard(result))
        //console.log(result)

    }
      
})

function CreateCard(result)
{
    const card = document.createElement('div');
    card.classList.add("card")
    const cardBody = document.createElement('div');
    cardBody.classList.add("card-body")
    const cardTitle = document.createElement('h5');
    cardTitle.innerHTML = `Email: ${result.id}`
    cardTitle.classList.add("card-title")
    const cardContent = document.createElement('p');
    cardContent.innerHTML = `Elapsed time (seconds): ${result.elapsed_time}`

    result.face.forEach(face => 
    {
        newImg = document.createElement('img');

        // Set the src and alt attributes for the new img element
        newImg.setAttribute('src', `data:image/png;base64,${face.base64}`);
        newImg.setAttribute('alt', 'Image description');
        card.appendChild(newImg)
    });

    cardBody.appendChild(cardTitle)
    cardBody.appendChild(cardContent)
    card.appendChild(cardBody)
    return card
}

function CreateErrorMsgCard(result)
{
    const card = document.createElement('div');
    card.classList.add("card")
    const cardBody = document.createElement('div');
    cardBody.classList.add("card-body")
    const cardTitle = document.createElement('h5');
    cardTitle.innerHTML = `An Unexpected Error`
    cardTitle.classList.add("card-title")
    const cardContent = document.createElement('p');
    cardContent.innerHTML = `${result}`

   
    cardBody.appendChild(cardTitle)
    cardBody.appendChild(cardContent)
    card.appendChild(cardBody)
    return card
}