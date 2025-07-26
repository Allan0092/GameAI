using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class hitpoints_txt_follow_main_camera : MonoBehaviour
{
    
    void LateUpdate()
    {
        if (Camera.main != null)
        {
            // Get the text size
            float textSizeX = GetComponent<Renderer>().bounds.size.x;
            float textSizeY = GetComponent<Renderer>().bounds.size.y;

            // Get the viewport coordinates for the top-left corner
            Vector3 viewportPos = new Vector3(0, 1, Camera.main.nearClipPlane);

            // Convert viewport coordinates to world coordinates
            Vector3 worldPos = Camera.main.ViewportToWorldPoint(viewportPos);

            // Adjust the position based on the text size
            float adjustedX = worldPos.x + textSizeX / 2f;
            float adjustedY = worldPos.y - textSizeY / 2f;

            // Update the position of the text
            transform.position = new Vector3(adjustedX, adjustedY, transform.position.z);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
    
    }
}
