// Sidebar Navigation

const links = document.querySelectorAll(".nav-link");
const screens = document.querySelectorAll(".screen");

links.forEach(link => {

  link.addEventListener("click", () => {

    // Remove active class
    links.forEach(item => {
      item.classList.remove("active");
    });

    // Add active class
    link.classList.add("active");

    // Hide screens
    screens.forEach(screen => {
      screen.classList.remove("active-screen");
    });

    // Show selected screen
    const screenId = link.getAttribute("data-screen");

    document
      .getElementById(screenId)
      .classList.add("active-screen");

  });

});

// Image Preview

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("previewImage");

imageInput.addEventListener("change", function(e){

  const file = e.target.files[0];

  if(file){

    preview.style.display = "block";

    previewImage.src = URL.createObjectURL(file);

  }

});

// Chart

const ctx = document.getElementById("analyticsChart");

new Chart(ctx, {

  type:"line",

  data:{

    labels:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],

    datasets:[{

      label:"AI Analytics",

      data:[12,19,8,15,25,18,30],

      borderWidth:3,
      tension:0.4

    }]

  },

  options:{

    responsive:true,

    plugins:{
      legend:{
        labels:{
          color:"white"
        }
      }
    },

    scales:{

      y:{
        ticks:{
          color:"white"
        }
      },

      x:{
        ticks:{
          color:"white"
        }
      }

    }

  }

});