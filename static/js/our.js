AOS.init()
AOS.init({
    // Settings that can be overridden on per-element basis, by `data-aos-*` attributes:
    offset: 120, // offset (in px) from the original trigger point
    delay: 0, // values from 0 to 3000, with step 50ms
    duration: 700, // values from 0 to 3000, with step 50ms
    easing: 'ease', // default easing for AOS animations
    once: false, // whether animation should happen only once - while scrolling down
    mirror: false, // whether elements should animate out while scrolling past them
    anchorPlacement: 'top-bottom', // defines which position of the element regarding to window should trigger the animation
  
  });   
   const change1 = document.getElementById("change-link1")
    const change2 = document.getElementById("change-link2")
    // togswitch.addEventListener("click",handleClickEvent(this),false);
    function handleChangeEventDetect(el){
        if(el.getAttribute("aria-checked")==="true"){
           change1.innerHTML = "Detect Alzheimers";
           el.setAttribute("aria-checked","false") ;
           change1.setAttribute("href","./alzheimersDetect.html")
        }else{
            change1.innerHTML = " Detect Tumor";
            el.setAttribute("aria-checked","true") ;
            change1.setAttribute("href","./brainTumorDetect.html")
        }
    }
    function handleChangeEventClass(el){
        if(el.getAttribute("aria-checked")==="true"){
           change2.innerHTML = "Classify Alzheimers";
           el.setAttribute("aria-checked","false") ;
           change2.setAttribute("href","./test.html")
        }else{
            change2.innerHTML = " Classify Tumor";
            el.setAttribute("aria-checked","true") ;
            change2.setAttribute("href","./slider.html")
        }
    }