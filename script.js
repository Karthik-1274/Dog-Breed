$(document).ready(function() {
    $("#fileInput").change(function(event) {
        let file = event.target.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = function(e) {
                $("#previewImg").attr("src", e.target.result).removeClass("hidden");
            };
            reader.readAsDataURL(file);
        }
    });

    $("#uploadBtn").click(function() {
        let file = $("#fileInput")[0].files[0];
        if (!file) {
            alert("⚠️ Please select an image file first!");
            return;
        }

        let formData = new FormData();
        formData.append("file", file);

        $("#loading").removeClass("hidden");
        $("#result").html("");

        $.ajax({
            url: "http://127.0.0.1:5000/predict",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $("#loading").addClass("hidden");
                $("#result").html(
                    `<p>🐾 Breed: <span style="color:#FFD700;">${response.breed}</span></p>
                     <p>🔍 Confidence: <span style="color:#00FF00;">${(response.confidence * 100).toFixed(2)}%</span></p>`
                );
            },
            error: function() {
                $("#loading").addClass("hidden");
                $("#result").html("<p class='text-danger'>❌ Error predicting breed. Ensure Flask server is running.</p>");
            }
        });
    });
});
