 function getSuggestions() {
  const input = document.getElementById('film_name');
  const datalist = document.getElementById('suggestionsList');
  const inputValue = input.value;

  datalist.innerHTML = '';

  fetch(`/api/film_suggestions?film_name=${inputValue}`)
    .then(response => response.json())
    .then(data => {
      data.suggestions.forEach(film => {
        const option = document.createElement('option');
        option.value = film;
        datalist.appendChild(option);
      });
    });
}
var modal = document.getElementById("myModal");
modal.style.display = "none";
function myFunction() {
  var modal = document.getElementById("myModal");
  var span = document.getElementsByClassName("close")[0];
  modal.style.display = "block";
  span.onclick = function() {
    modal.style.display = "none";
  }
  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
}

