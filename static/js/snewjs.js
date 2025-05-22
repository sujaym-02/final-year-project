$(document).ready(function () {
    // Init
    // 
    // const segButn = document.createElement("button");
    // segButn.innerHTML = "Result Preview";
    // segButn.setAttribute("id","resButn");
    // $("btn-predict").append(segButn);

    // const resultPreview = document.createElement("img");
    // resultPreview.setAttribute("id","resultPreview");
    // resultPreview.setAttribute("src","#");
    // $('.image-section').append(resultPreview)

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr( 'src', e.target.result );
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#resultPreview').hide();
        $('#btn-predict').show();
        $('#btn-result').hide();
        // $('#btn-resultcls').hide();
        $('#result').text('');
        $('#result').hide();
        // $('#segButn').hide();
        readURL(this);
    });
    // $('#btn-result').click(function(){
    //     $("#resultPreview").show();
    //     $(this).hide();
    // });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/segment_bt',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#btn-result').show();
                $('#result').text('Result Saved successfully');
                $('#result').fadeIn(600);
                console.log('Success!');
        
            },
        });
    });

    
    $('#btn-result').click(function(){
        $(this).hide();
    $.ajax({
        url:'/test',
        type:'GET',
        contentType: "image/png",
        success: function(result)
        {
            path = 'data:;base64,' + result['image']
            // console.log(path)
            $('#resultPreview').attr('src',path)
            $('#resultPreview').show()

        }});
    });

});
