namespace Marmotte

open System
open Microsoft.AspNetCore.Builder

module RestfulDsl =

    type RouteResultDefinition =
        { Description: string
          StatusCode: int
          Model: Type }
    
    [<RequireQualifiedAccess>]
    module RouteResultDefinition =
        let create<'t> statusCode desc =
            { StatusCode=statusCode; Description=desc; Model=typeof<'t> }

    type RestOperationConfig =
        { OperationName: string option
          ConfigureRoutes: WebApplication -> unit
          Results: RouteResultDefinition list }
    
    [<RequireQualifiedAccess>]
    module RestOperationConfig =
        let addResponse (result:RouteResultDefinition) (operation:RestOperationConfig) =
            { operation with Results = result :: operation.Results  }

    type RestResource =
        { Name: string; Operations: RestOperationConfig list }
    
    [<RequireQualifiedAccess>]    
    module RestResource =
        let addResponse (result:RouteResultDefinition) (resource:RestResource) =
            let operations = 
                match resource.Operations with
                | [] -> []
                | [operation] -> [operation |> RestOperationConfig.addResponse result]
                | operation :: xs -> (operation |> RestOperationConfig.addResponse result) :: xs 
            { resource with Operations=operations }
    
    type RestApiDefinition =
        { Resources: RestResource list
          App: WebApplication }
    
    [<RequireQualifiedAccess>]    
    module RestApiDefinition =
        let addResponse (result:RouteResultDefinition) (def:RestApiDefinition) =
           let resources =
               match def.Resources with
               | [resource] -> [resource |> RestResource.addResponse result]
               | resource :: xs -> (resource |> RestResource.addResponse result) :: xs
               | [] -> []
           { def with Resources=resources }
    
        let create app =
           { App=app; Resources=[ { Name=String.Empty; Operations=[] } ] }

        let currentResourceName def =
           def.Resources |> List.head |> _.Name
    
    type RestfulBuilder(app: RestApiDefinition) =
        
        member _.Yield(state:RestApiDefinition) = state
        
        member _.Yield(()) = app

        member _.Run _ =
            // TODO: run route configuration functions
            app.App.Run()
        
        // member _.For(state:RestApiDefinition, f: RestOperationConfig -> RestOperationConfig) =
        //     state
        //     
        // member _.Combine(state:RestApiDefinition, c:RestOperationConfig) =
        //     state
        
        // member _.Return (state:RestApiDefinition, c:RestOperationConfig) =
        //     state
        
        member _.Delay (func:unit->RestApiDefinition) =
            func()
      
        [<CustomOperation("replies")>]
        member x.Replies (state:RestApiDefinition, rs:RouteResultDefinition) =
            state |> RestApiDefinition.addResponse rs

        [<CustomOperation("resource")>]
        member x.Resource (state:RestApiDefinition, resourceName: string) =
            { state with Resources={ Name=resourceName; Operations=[] } :: state.Resources }
        
        [<CustomOperation("get")>]
        member x.Map<'tin, 'tout, 'terr> (state:RestApiDefinition, verb, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.map verb routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("get")>]
        member x.Get<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.get routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("post")>]
        member x.Post<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.post routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("delete")>]
        member x.Delete<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.delete routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("patch")>]
        member x.Patch<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.patch routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("options")>]
        member x.Options<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.options routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("put")>]
        member x.Put<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.put routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
        
        [<CustomOperation("trace")>]
        member x.Trace<'tin, 'tout, 'terr> (state:RestApiDefinition, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            // TODO: update state instead of create route directly
            let resourceName = state |> RestApiDefinition.currentResourceName
            Restful.trace routeTemplate handler id { App=state.App; Name=resourceName; Tags=resourceName } |> ignore
            state
    
    let restful app =
        let builder = app |> RestApiDefinition.create |> RestfulBuilder
        builder
    
    
