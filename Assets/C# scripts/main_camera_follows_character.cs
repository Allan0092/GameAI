// Moving the camera to match the character's position. This C# script should be attached to the 'Main Camera'
// GameObject in the Unity(C#) software/editor

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class main_camera_follows_character : MonoBehaviour
{
    // Setting a reference to the character GameObject's 'Transform' component
    private Transform target;

    private void Awake()
    {
        // Getting a reference to the character's 'Transform' component by specifying their tag here 
        // (you need to manually set the tag for the character GameObject in the Unity(C#) software/editor
        // in order for this to work).

        //"PLayer" is the name of the tag of the character GameObject in Unity(C#) software/editor
        target = GameObject.FindWithTag("Player").transform;
    }

    private void LateUpdate()
    {   
        // Setting the 'cameraPosition' variable to be the position of the 'Main Camera' GameObject this
        // C# script is attached to (by calling the 'position' property of the 'Main Camera' GameObject's 
        // 'Transform' component)
        Vector3 cameraPosition = transform.position;

        // Only changing the camera's position in the x-axis direction w.r.t to the character's position
        cameraPosition.x = target.position.x;
        transform.position = cameraPosition;
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
